"""CLI: ``imagegen generate|edit|upscale|seamless|key|seam-score|backends|models``, ``imagegen atlas ...``
(grid texture atlases) and ``imagegen uv ...`` (retexturing a Blender object's UV atlas).

Every command prints a JSON summary with ``--json`` (for agents/scripts); progress goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from imagegen import atlas, blender_bridge, ops, pipeline, uvtex
from imagegen.backends import BACKENDS, ImageGenError, backend_status, get_backend


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _backend(args: argparse.Namespace):
    return get_backend(args.backend, args.model)


def _emit(args: argparse.Namespace, data: dict[str, Any], text: str) -> None:
    print(json.dumps(data, indent=2, default=str) if args.json else text)


def _out_paths(out: str, n: int) -> list[Path]:
    p = Path(out)
    return [p] if n == 1 else [p.with_name(f"{p.stem}-{i + 1}{p.suffix or '.png'}") for i in range(n)]


def _region(args: argparse.Namespace, size: tuple[int, int]):
    return ops.parse_region(args.region, size) if getattr(args, "region", None) else None


# --------------------------------------------------------------------------- #
# single-image commands
# --------------------------------------------------------------------------- #


def cmd_generate(args: argparse.Namespace) -> int:
    backend = _backend(args)
    _log(f"generating with {backend.name}:{backend.model} ...")
    images, res = pipeline.generate(
        args.prompt,
        backend=backend,
        references=args.ref or [],
        size=ops.parse_size(args.size) if args.size else None,
        aspect_ratio=args.aspect,
        n=args.n,
        fit=args.fit,
    )
    paths = [ops.save_image(im, p) for im, p in zip(images, _out_paths(args.out, len(images)))]
    if args.transparent:
        paths = [ops.save_image(ops.chroma_key(ops.load_image(p), ops.parse_color(args.transparent)), p) for p in paths]
    _emit(
        args,
        {"outputs": [str(p) for p in paths], "size": list(images[0].size), "backend": res.backend, "model": res.model,
         "usage": res.usage, **res.meta},  # fmt: skip
        "\n".join(f"wrote {p} {images[0].size[0]}x{images[0].size[1]}" for p in paths),
    )
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    base = ops.load_image(args.image)
    backend = _backend(args)
    res = pipeline.edit(
        base,
        args.instruction,
        backend=backend,
        region=_region(args, base.size),
        mask=args.mask,
        context=args.context,
        feather=args.feather,
        references=args.ref or [],
        color_match=not args.no_color_match,
        log=_log,
    )
    out = ops.save_image(res.image, args.out or args.image)
    _emit(args, {"output": str(out), **res.info()}, f"wrote {out}")
    return 0


def cmd_upscale(args: argparse.Namespace) -> int:
    backend = _backend(args) if args.method == "ai" else None
    img = pipeline.upscale(args.image, args.scale, method=args.method, backend=backend, tile=args.tile,
                           overlap=args.overlap, hint=args.hint or "", concurrency=args.concurrency, log=_log)  # fmt: skip
    out = ops.save_image(img, args.out)
    _emit(args, {"output": str(out), "size": list(img.size), "method": args.method}, f"wrote {out} {img.size[0]}x{img.size[1]}")
    return 0


def cmd_seamless(args: argparse.Namespace) -> int:
    before = ops.seam_score(ops.load_image(args.image))
    backend = _backend(args) if args.method == "ai" else None
    img = pipeline.seamless(args.image, method=args.method, backend=backend, band=args.band, hint=args.hint or "", log=_log)
    out = ops.save_image(img, args.out)
    if args.preview:
        ops.save_image(ops.tile_preview(img, 2), args.preview)
    after = ops.seam_score(img)
    _emit(args, {"output": str(out), "seam_score_before": round(before, 3), "seam_score_after": round(after, 3)},
          f"wrote {out}  seam score {before:.2f} -> {after:.2f} (~1.0 = invisible)")  # fmt: skip
    return 0


def cmd_seam_score(args: argparse.Namespace) -> int:
    scores = {p: round(ops.seam_score(ops.load_image(p)), 3) for p in args.images}
    _emit(args, {"seam_scores": scores, "threshold": atlas.SEAM_THRESHOLD},
          "\n".join(f"{s:6.2f}  {p}" for p, s in scores.items()))  # fmt: skip
    return 0


def cmd_key(args: argparse.Namespace) -> int:
    img = ops.chroma_key(ops.load_image(args.image), ops.parse_color(args.color), args.tolerance, args.softness)
    out = ops.save_image(img, args.out)
    _emit(args, {"output": str(out)}, f"wrote {out}")
    return 0


def cmd_backends(args: argparse.Namespace) -> int:
    st = backend_status()
    lines = [f"default backend: {st['default']}"]
    c = st["codex"]
    lines.append(f"codex:      {'installed' if c['installed'] else 'NOT installed'}  model={c['model']}  "
                 f"{c.get('version', '')}  {c.get('login', '')}")  # fmt: skip
    o = st["openrouter"]
    lines.append(f"openrouter: key {'set' if o['api_key_set'] else 'NOT set'}  model={o['model']}  judge={o['judge_model']}")
    _emit(args, st, "\n".join(lines))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    # Discovery is public on OpenRouter; no key needed.
    import httpx

    data = httpx.get("https://openrouter.ai/api/v1/images/models", timeout=30).json().get("data", [])
    rows = [{"id": m["id"], "inputs": m.get("architecture", {}).get("input_modalities"),
             "params": sorted((m.get("supported_parameters") or {}).keys())} for m in data]  # fmt: skip
    _emit(args, {"models": rows}, "\n".join(f"{r['id']:<45} in={','.join(r['inputs'] or [])}  {' '.join(r['params'])}" for r in rows))
    return 0


# --------------------------------------------------------------------------- #
# atlas commands
# --------------------------------------------------------------------------- #


def cmd_atlas_init(args: argparse.Namespace) -> int:
    path = Path(args.spec)
    if path.exists() and not args.force:
        raise ImageGenError(f"{path} exists (use --force to overwrite)")
    example = atlas.EXAMPLE_SPRITES if args.example == "sprites" else atlas.EXAMPLE_SPEC
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(example, indent=2) + "\n")
    _emit(args, {"spec": str(path)}, f"wrote {path} — edit the cells, then: imagegen atlas build {path} -o <dir>")
    return 0


def cmd_atlas_build(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec)
    spec = atlas.AtlasSpec.load(spec_path / "spec.json" if spec_path.is_dir() else spec_path)
    outdir = args.out or (str(spec_path) if spec_path.is_dir() else str(spec_path.with_suffix("")))
    if args.mode:
        spec.mode = args.mode
    res = atlas.build(spec, outdir, _backend(args), only=_names(args.only), force=args.force,
                      concurrency=args.concurrency, log=_log)  # fmt: skip
    text = f"atlas {res['atlas']} {res['size'][0]}x{res['size'][1]} ({res['grid'][0]}x{res['grid'][1]})"
    if res["errors"]:
        text += "\nfailed: " + ", ".join(f"{k} ({v})" for k, v in res["errors"].items())
    _emit(args, res, text)
    return 1 if res["errors"] else 0


def cmd_atlas_cell(args: argparse.Namespace) -> int:
    proj = atlas.AtlasProject(args.dir)
    spec = proj.spec()
    region = ops.parse_region(args.region, spec.cell_size) if args.region else None
    needs_backend = args.edit is not None or args.prompt is not None
    res = atlas.edit_cell(
        args.dir, args.name, _backend(args) if needs_backend else None,
        instruction=args.edit, prompt=args.prompt, image=args.image, region=region, mask=args.mask,
        references=args.ref or [], log=_log,
    )  # fmt: skip
    _emit(args, res, f"updated {res['path']} (history: {res['history']}), repacked {res['atlas']}")
    return 0


def cmd_atlas_revert(args: argparse.Namespace) -> int:
    res = atlas.revert_cell(args.dir, args.name, args.steps)
    _emit(args, res, f"restored {args.name} from {res['restored']}")
    return 0


def cmd_atlas_repack(args: argparse.Namespace) -> int:
    proj = atlas.AtlasProject(args.dir)
    if args.padding is not None or args.columns is not None or args.pot:
        spec = proj.spec()
        spec.padding = spec.padding if args.padding is None else args.padding
        spec.columns = spec.columns if args.columns is None else args.columns
        spec.power_of_two = spec.power_of_two or args.pot
        spec.save(proj.spec_path)
    res = atlas.repack(args.dir)
    _emit(args, res, f"atlas {res['atlas']} {res['size'][0]}x{res['size'][1]}" + (f"  missing: {', '.join(res['missing'])}" if res["missing"] else ""))
    return 0


def cmd_atlas_review(args: argparse.Namespace) -> int:
    res = atlas.review(args.dir, _backend(args), fix=args.fix, threshold=args.threshold, rounds=args.rounds,
                       consistency=not args.no_consistency, only=_names(args.only), concurrency=args.concurrency, log=_log)  # fmt: skip
    lines = []
    for name, r in res["cells"].items():
        if "error" in r:
            lines.append(f"  ERR  {name}: {r['error']}")
            continue
        seam = f" seam={r['seam_score']:.2f}" if r.get("seam_score") is not None else ""
        lines.append(f"  {r['score']:>3}  {name}{seam}  {'; '.join(r['issues'][:2])}")
    if res.get("style_outliers"):
        lines.append("style outliers: " + ", ".join(f"{o['name']} ({o['reason']})" for o in res["style_outliers"]))
    lines.append(f"report: {res['path']}")
    _emit(args, res, "\n".join(lines))
    return 0


def cmd_atlas_upscale(args: argparse.Namespace) -> int:
    backend = _backend(args) if args.method == "ai" else None
    res = atlas.upscale_atlas(args.dir, args.scale, method=args.method, backend=backend, concurrency=args.concurrency, log=_log)
    _emit(args, res, f"atlas {res['atlas']} {res['size'][0]}x{res['size'][1]}")
    return 0


def cmd_atlas_import(args: argparse.Namespace) -> int:
    res = atlas.import_atlas(
        args.image, ops.parse_grid(args.grid), args.out, names=_names(args.names), source_padding=args.source_padding,
        padding=args.padding, style=args.style or "", tileable=args.tileable,
        background="transparent" if args.transparent else "opaque",
    )  # fmt: skip
    _emit(args, res, f"imported {res['cells']} cells into {args.out}; edit {args.out}/spec.json to describe them")
    return 0


def cmd_atlas_slice(args: argparse.Namespace) -> int:
    paths = atlas.slice_image(args.image, ops.parse_grid(args.grid), args.out, names=_names(args.names),
                              source_padding=args.source_padding)  # fmt: skip
    _emit(args, {"cells": [str(p) for p in paths]}, f"wrote {len(paths)} cells to {args.out}")
    return 0


def cmd_atlas_pack(args: argparse.Namespace) -> int:
    res = atlas.pack_images(args.images, args.out, columns=args.columns, padding=args.padding,
                            cell_size=ops.parse_size(args.cell) if args.cell else None, tileable=args.tileable,
                            power_of_two=args.pot)  # fmt: skip
    _emit(args, res, f"wrote {res['atlas']} {res['size'][0]}x{res['size'][1]} + {res['json']}")
    return 0


def cmd_atlas_guide(args: argparse.Namespace) -> int:
    cols, rows = ops.parse_grid(args.grid)
    img, _ = ops.grid_guide(cols, rows, ops.parse_size(args.cell), _names(args.names))
    out = ops.save_image(img, args.out)
    _emit(args, {"output": str(out)}, f"wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
# uv (Blender UV atlas) commands
# --------------------------------------------------------------------------- #


def _notes(items: list[str] | None) -> dict[int, str] | None:
    """``--note 3="roof shingles"`` (repeatable) -> {3: "roof shingles"}."""
    if not items:
        return None
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not v:
            raise ValueError(f"Bad --note {it!r}; expected ISLAND=description")
        out[int(k)] = v.strip()
    return out


def _suffix(path: str | Path, tag: str) -> Path:
    p = Path(path)
    return p.with_name(f"{p.stem}.{tag}{p.suffix or '.png'}")


def _facing(uv: uvtex.UVLayout, k: int) -> dict[str, float]:
    """Fraction of island k's texels matched by the up/side/down selectors (the island's average
    normal can mislead: a half-wall, half-roof island averages to 'slightly up')."""
    isl = uv.islands == k
    n = max(int(isl.sum()), 1)
    return {sel: round(float((uvtex.select_texels(uv, sel) & isl).sum() / n), 3) for sel in ("up", "side", "down")}


def _uv_summary(uv: uvtex.UVLayout) -> dict[str, Any]:
    info = uv.info
    return {
        "object": info.get("object"), "uv_map": info.get("uv_map"), "size": info["size"],
        "islands": [{**{k: i.get(k) for k in ("id", "bbox", "texels", "normal", "up_rotation", "texel_density")},
                     **({"faces": _facing(uv, i["id"])} if uv.normals is not None else {})} for i in info["islands"]],
        "seams": len(info.get("seams", [])), "textures": info.get("textures", []), "sanity": info.get("sanity", {}),
    }  # fmt: skip


def cmd_uv_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    scene = blender_bridge.make_demo_scene(out, args.size)
    blender_bridge.export_uv(scene["blend"], out / "uv")
    uv = uvtex.UVLayout.load(out / "uv")
    uvtex.placeholder(uv).save(scene["atlas"])
    uv.preview().save(out / "uv" / "uv_preview.png")
    _emit(args, {**scene, "uv": str(out / "uv")},
          f"demo scene {scene['blend']} (object {scene['object']}), placeholder atlas {scene['atlas']}, UV data {out / 'uv'}")  # fmt: skip
    return 0


def cmd_uv_export(args: argparse.Namespace) -> int:
    blender_bridge.export_uv(args.blend, args.out, obj=args.object, uv_map=args.uv_map, size=args.size)
    uv = uvtex.UVLayout.load(args.out)
    uv.preview().save(Path(args.out) / "uv_preview.png")
    Image.fromarray((uv.mask * 255).astype("uint8"), "L").save(Path(args.out) / "uv_mask.png")
    summary = _uv_summary(uv)
    tex = summary["textures"]
    lines = [f"{summary['object']} uv={summary['uv_map']} {summary['size'][0]}x{summary['size'][1]}: "
             f"{len(summary['islands'])} islands, {summary['seams']} seams"]
    for i in summary["islands"]:
        lines.append(f"  island {i['id']}: {i['texels']} texels, faces {i.get('faces')}, up points {i.get('up_rotation')} deg")
    lines += [
             "textures: " + (", ".join(f"{t['material']}/{t['node']} -> {t['filepath']} ({t['colorspace']})" for t in tex) or "none"),
             f"sanity: {summary['sanity']}",
             f"preview: {Path(args.out) / 'uv_preview.png'}"]  # fmt: skip
    _emit(args, summary, "\n".join(lines))
    return 0


def cmd_uv_placeholder(args: argparse.Namespace) -> int:
    uv = uvtex.UVLayout.load(args.uv)
    out = ops.save_image(uvtex.placeholder(uv, args.base), args.out)
    _emit(args, {"output": str(out)}, f"wrote {out}")
    return 0


def _print_check(rep: dict[str, Any]) -> str:
    lines = [f"{'PASS' if rep['passed'] else 'FAIL'}  errors={rep['errors']} warnings={rep['warnings']}"]
    for c in rep["checks"]:
        mark = "ok  " if c["passed"] else ("ERR " if c["severity"] == "error" else "warn")
        lines.append(f"  {mark} {c['name']}: {c['detail']}")
    if rep["repair_islands"]:
        lines.append(f"  repair islands: {rep['repair_islands']}")
    return "\n".join(lines)


def cmd_uv_retexture(args: argparse.Namespace) -> int:
    uv = uvtex.UVLayout.load(args.uv)
    atlas_path = args.atlas or next((t["filepath"] for t in uv.info.get("textures", []) if t.get("filepath")), None)
    if not atlas_path:
        raise ValueError("No atlas given and none recorded in uv_info.json")
    backend = _backend(args)
    notes = _notes(args.note)
    materials = uvtex.parse_materials(args.material)
    instruction = args.instruction or ""
    res = uvtex.retexture(atlas_path, uv, args.style, backend=backend, instruction=instruction, island_notes=notes,
                          materials=materials, padding=args.padding, mode=args.mode, tile=args.tile, log=_log)  # fmt: skip
    final, raw = res.image, res.raw
    ops.save_image(res.guide, _suffix(args.out, "guide"))
    rep: dict[str, Any] = {}
    history = []
    for rnd in range(args.repair_rounds + 1):
        final, fin = uvtex.finish(final, atlas_path, uv, materials=materials, padding=args.padding, raw=raw,
                                  seams=not args.no_fix_seams, flatten=args.flatten_lighting and rnd == 0)  # fmt: skip
        if fin["filled_islands"]:
            _log(f"filled small unpainted strips on islands {fin['filled_islands']} (fraction of island)")
        rep = uvtex.check(final, atlas_path, uv, raw=raw, padding=args.padding, style=args.style[0] if args.judge else None,
                          backend=backend if args.judge else None, materials=materials, instruction=instruction)  # fmt: skip
        history.append({"round": rnd + 1, "passed": rep["passed"], "errors": rep["errors"], "warnings": rep["warnings"],
                        "repair_islands": rep["repair_islands"]})  # fmt: skip
        _log(f"check round {rnd + 1}: {'PASS' if rep['passed'] else 'FAIL'} errors={rep['errors']} warnings={rep['warnings']}"
             + (f" -> repair islands {rep['repair_islands']}" if rep["repair_islands"] and rnd < args.repair_rounds else ""))  # fmt: skip
        if rep["passed"] or not rep["repair_islands"] or rnd == args.repair_rounds:
            break
        final, raw = uvtex.repair(final, atlas_path, uv, rep["repair_islands"], args.style, backend=backend, raw=raw,
                                  instruction=instruction, island_notes=notes, materials=materials,
                                  padding=args.padding, log=_log)  # fmt: skip
    rep["rounds"] = history
    rep["groups"] = res.groups
    out = ops.save_image(final, args.out)
    ops.save_image(raw, _suffix(args.out, "raw"))
    _suffix(args.out, "check").with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    _emit(args, {"output": str(out), "raw": str(_suffix(args.out, "raw")), "check": rep, "calls": res.calls},
          f"wrote {out} (+ .raw.png, .guide.png, .check.json)\n" + _print_check(rep))  # fmt: skip
    return 0 if rep.get("passed") else 2


def cmd_uv_check(args: argparse.Namespace) -> int:
    uv = uvtex.UVLayout.load(args.uv)
    backend = _backend(args) if args.judge else None
    rep = uvtex.check(args.image, args.original, uv, raw=args.raw, padding=args.padding, style=args.style,
                      backend=backend, materials=uvtex.parse_materials(args.material), instruction=args.instruction or "")  # fmt: skip
    if args.report:
        Path(args.report).write_text(json.dumps(rep, indent=2, default=str) + "\n")
    _emit(args, rep, _print_check(rep))
    return 0 if rep["passed"] else 2


def cmd_uv_repair(args: argparse.Namespace) -> int:
    uv = uvtex.UVLayout.load(args.uv)
    islands = [int(k) for k in args.islands.split(",")] if args.islands else \
        json.loads(Path(args.from_check).read_text())["repair_islands"]  # fmt: skip
    if not islands:
        raise ValueError("No islands to repair")
    final, raw = uvtex.repair(args.image, args.original, uv, islands, args.style, backend=_backend(args), raw=args.raw,
                              instruction=args.instruction or "", island_notes=_notes(args.note),
                              materials=uvtex.parse_materials(args.material), padding=args.padding, log=_log)  # fmt: skip
    final, _ = uvtex.finish(final, args.original, uv, materials=uvtex.parse_materials(args.material), padding=args.padding, raw=raw)
    out = ops.save_image(final, args.out)
    ops.save_image(raw, _suffix(args.out, "raw"))
    rep = uvtex.check(final, args.original, uv, raw=raw, padding=args.padding, materials=uvtex.parse_materials(args.material))
    _emit(args, {"output": str(out), "repaired": islands, "check": rep}, f"wrote {out}\n" + _print_check(rep))
    return 0 if rep["passed"] else 2


def cmd_uv_bleed(args: argparse.Namespace) -> int:
    img = ops.load_image(args.image)
    uv = uvtex.UVLayout.load(args.uv).resized(img.size)
    out = ops.save_image(uvtex.bleed(img, uv.mask, args.padding), args.out)
    _emit(args, {"output": str(out)}, f"wrote {out}")
    return 0


def cmd_uv_fix_seams(args: argparse.Namespace) -> int:
    img = ops.load_image(args.image)
    uv = uvtex.UVLayout.load(args.uv).resized(img.size)
    fixed = uvtex.bleed(uvtex.fix_seams(img, uv, width=args.width, original=args.original), uv.mask, args.padding, keep=img)
    out = ops.save_image(fixed, args.out)
    _emit(args, {"output": str(out)}, f"wrote {out}")
    return 0


def cmd_uv_fill_holes(args: argparse.Namespace) -> int:
    img = ops.load_image(args.image)
    uv = uvtex.UVLayout.load(args.uv)
    fixed, info = uvtex.finish(img, args.original, uv, materials=uvtex.parse_materials(args.material),
                               padding=args.padding, max_fill=args.max_fill, seams=False, raw=args.raw)  # fmt: skip
    out = ops.save_image(fixed, args.out)
    _emit(args, {"output": str(out), **info}, f"wrote {out}; filled islands {info['filled_islands'] or 'none'}")
    return 0


def cmd_uv_flatten(args: argparse.Namespace) -> int:
    img = ops.load_image(args.image)
    uv = uvtex.UVLayout.load(args.uv)
    fixed, _ = uvtex.finish(img, args.original, uv, materials=uvtex.parse_materials(args.material), padding=args.padding,
                            fill=False, seams=False, flatten=True)  # fmt: skip
    out = ops.save_image(fixed, args.out)
    _emit(args, {"output": str(out)}, f"wrote {out}")
    return 0


def cmd_uv_render(args: argparse.Namespace) -> int:
    res = blender_bridge.render(args.blend, args.out, image=args.image, obj=args.object, replace=args.replace,
                                views=args.views, res=args.res, engine=args.engine, samples=args.samples, save=args.save,
                                opaque=args.opaque)  # fmt: skip
    _emit(args, res, "rendered " + ", ".join(res["renders"]) + (f"\nswapped: {res['swapped']}" if res["swapped"] else ""))
    return 0


def cmd_uv_review_renders(args: argparse.Namespace) -> int:
    def pngs(d: str | None) -> list[Path]:
        return sorted(Path(d).glob("*.png")) if d else []

    after = pngs(args.after)
    if not after:
        raise ValueError(f"No renders in {args.after}")
    verdict = uvtex.review_renders(after, args.style, backend=_backend(args), before=pngs(args.before))
    text = [f"style_match={verdict['style_match']} improved={verdict['improved']} seams={verdict['visible_seams']} "
            f"stretch={verdict['stretching_or_misalignment']} lighting={verdict['painted_lighting']} "
            f"pattern={verdict['pattern_scale_or_direction_problems']}"]  # fmt: skip
    text += [f"  - {i}" for i in verdict["issues"]] + [f"  fix: {f}" for f in verdict["fix_suggestions"]]
    _emit(args, verdict, "\n".join(text))
    return 0


def _names(text: str | None) -> list[str] | None:
    return [n.strip() for n in text.split(",") if n.strip()] if text else None


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", "-b", choices=list(BACKENDS), help="codex (ChatGPT sub, default when installed) or openrouter")
    common.add_argument("--model", "-m", help="Backend model (codex: gpt-5.5; openrouter: e.g. google/gemini-3.1-flash-image)")
    common.add_argument("--json", action="store_true", help="Print a JSON summary")

    p = argparse.ArgumentParser(prog="imagegen", description="Generate, edit, upscale and atlas images with AI image models")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", parents=[common], help="Text (+ reference images) -> image")
    g.add_argument("prompt")
    g.add_argument("-o", "--out", required=True, help="Output path (several images get -1, -2 suffixes)")
    g.add_argument("--ref", "-r", action="append", help="Reference image (repeatable)")
    g.add_argument("--size", help="Exact output size, N or WxH (resized/cropped after generation)")
    g.add_argument("--aspect", help="Aspect ratio hint, e.g. 1:1, 3:2, 16:9")
    g.add_argument("--fit", default="cover", choices=["cover", "contain", "stretch"], help="How --size is applied")
    g.add_argument("-n", type=int, default=1, help="Number of images")
    g.add_argument("--transparent", metavar="KEYCOLOR", help="Chroma-key this colour to alpha afterwards (e.g. '#ff00ff')")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("edit", parents=[common], help="Edit an image, or only a region/mask of it")
    e.add_argument("image")
    e.add_argument("instruction")
    e.add_argument("-o", "--out", help="Output path (default: overwrite input)")
    e.add_argument("--region", help="x,y,w,h in pixels, or fractions like 0.25,0.25,0.5,0.5")
    e.add_argument("--mask", help="Mask image: white/opaque = editable")
    e.add_argument("--context", type=float, default=0.5, help="Surrounding context shown to the model, as a fraction of the region size")
    e.add_argument("--feather", type=float, help="Blend radius in px at the region edge (default: auto)")
    e.add_argument("--ref", "-r", action="append", help="Extra reference image (repeatable)")
    e.add_argument("--no-color-match", action="store_true", help="Don't correct the model's colour drift")
    e.set_defaults(func=cmd_edit)

    u = sub.add_parser("upscale", parents=[common], help="Enlarge an image (local resample or AI re-detail)")
    u.add_argument("image")
    u.add_argument("-o", "--out", required=True)
    u.add_argument("--scale", "-s", type=float, default=2.0)
    u.add_argument("--method", default="lanczos", choices=["lanczos", "bicubic", "nearest", "ai"])
    u.add_argument("--tile", type=int, default=1024, help="AI tile size in output pixels")
    u.add_argument("--overlap", type=int, default=128)
    u.add_argument("--hint", help="What the image is, to guide AI detail")
    u.add_argument("--concurrency", type=int, default=4)
    u.set_defaults(func=cmd_upscale)

    s = sub.add_parser("seamless", parents=[common], help="Make a texture tile seamlessly")
    s.add_argument("image")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--method", default="blend", choices=["blend", "ai"])
    s.add_argument("--band", type=float, default=0.2, help="Width of the blended/repainted band (fraction)")
    s.add_argument("--hint", help="Extra guidance for the AI repaint")
    s.add_argument("--preview", help="Also write a 2x2 tiled preview here")
    s.set_defaults(func=cmd_seamless)

    ss = sub.add_parser("seam-score", parents=[common], help="Measure wrap-around seam visibility (~1.0 = seamless)")
    ss.add_argument("images", nargs="+")
    ss.set_defaults(func=cmd_seam_score)

    k = sub.add_parser("key", parents=[common], help="Chroma-key a solid background colour to transparency")
    k.add_argument("image")
    k.add_argument("-o", "--out", required=True)
    k.add_argument("--color", default="#ff00ff")
    k.add_argument("--tolerance", type=float, default=60)
    k.add_argument("--softness", type=float, default=40)
    k.set_defaults(func=cmd_key)

    b = sub.add_parser("backends", parents=[common], help="Show which backends are usable")
    b.set_defaults(func=cmd_backends)
    m = sub.add_parser("models", parents=[common], help="List OpenRouter image models and their parameters")
    m.set_defaults(func=cmd_models)

    # ---- atlas ----
    a = sub.add_parser("atlas", help="Texture atlas projects").add_subparsers(dest="atlas_command", required=True)

    ai = a.add_parser("init", parents=[common], help="Write an example atlas spec")
    ai.add_argument("spec")
    ai.add_argument("--example", default="textures", choices=["textures", "sprites"])
    ai.add_argument("--force", action="store_true")
    ai.set_defaults(func=cmd_atlas_init)

    ab = a.add_parser("build", parents=[common], help="Generate missing cells and pack the atlas (resumable)")
    ab.add_argument("spec", help="spec.json, or an existing atlas project dir")
    ab.add_argument("-o", "--out", help="Project dir (default: spec path without .json)")
    ab.add_argument("--only", help="Comma-separated cell names")
    ab.add_argument("--force", action="store_true", help="Regenerate cells that already exist")
    ab.add_argument("--mode", choices=["cells", "sheet"], help="Override spec mode")
    ab.add_argument("--concurrency", type=int, default=4)
    ab.set_defaults(func=cmd_atlas_build)

    ac = a.add_parser("cell", parents=[common], help="Edit / regenerate / replace one cell, then repack")
    ac.add_argument("dir")
    ac.add_argument("name")
    grp = ac.add_mutually_exclusive_group(required=True)
    grp.add_argument("--edit", help="Edit instruction applied to the current cell")
    grp.add_argument("--prompt", help="New description; regenerates the cell")
    grp.add_argument("--image", help="Replace the cell with this image file")
    ac.add_argument("--region", help="With --edit: x,y,w,h (px or fractions) inside the cell")
    ac.add_argument("--mask", help="With --edit: mask image (white = editable)")
    ac.add_argument("--ref", "-r", action="append", help="Extra reference image for --edit")
    ac.set_defaults(func=cmd_atlas_cell)

    ar = a.add_parser("revert", parents=[common], help="Restore a cell's previous version")
    ar.add_argument("dir")
    ar.add_argument("name")
    ar.add_argument("--steps", type=int, default=1)
    ar.set_defaults(func=cmd_atlas_revert)

    ap = a.add_parser("repack", parents=[common], help="Rebuild atlas.png/json from cells/")
    ap.add_argument("dir")
    ap.add_argument("--padding", type=int)
    ap.add_argument("--columns", type=int)
    ap.add_argument("--pot", action="store_true", help="Pad to power-of-two size")
    ap.set_defaults(func=cmd_atlas_repack)

    av = a.add_parser("review", parents=[common], help="Vision-model QA of every cell; --fix repairs failures")
    av.add_argument("dir")
    av.add_argument("--fix", action="store_true")
    av.add_argument("--threshold", type=int, default=7, help="Scores below this fail (0-10)")
    av.add_argument("--rounds", type=int, default=2, help="Max fix/re-review rounds with --fix")
    av.add_argument("--only", help="Comma-separated cell names")
    av.add_argument("--no-consistency", action="store_true", help="Skip the whole-atlas style check")
    av.add_argument("--concurrency", type=int, default=4)
    av.set_defaults(func=cmd_atlas_review)

    au = a.add_parser("upscale", parents=[common], help="Upscale every cell and repack")
    au.add_argument("dir")
    au.add_argument("--scale", "-s", type=float, default=2.0)
    au.add_argument("--method", default="lanczos", choices=["lanczos", "bicubic", "nearest", "ai"])
    au.add_argument("--concurrency", type=int, default=4)
    au.set_defaults(func=cmd_atlas_upscale)

    am = a.add_parser("import", parents=[common], help="Turn an existing atlas image into an editable project")
    am.add_argument("image")
    am.add_argument("--grid", required=True, help="COLSxROWS")
    am.add_argument("-o", "--out", required=True)
    am.add_argument("--names", help="Comma-separated cell names, row-major")
    am.add_argument("--source-padding", type=int, default=0, help="Padding px to strip from each source slot")
    am.add_argument("--padding", type=int, default=0, help="Padding px in the repacked atlas")
    am.add_argument("--style", help="Art style description for later edits")
    am.add_argument("--tileable", action="store_true")
    am.add_argument("--transparent", action="store_true")
    am.set_defaults(func=cmd_atlas_import)

    asl = a.add_parser("slice", parents=[common], help="Cut a grid image into cell files")
    asl.add_argument("image")
    asl.add_argument("--grid", required=True, help="COLSxROWS")
    asl.add_argument("-o", "--out", required=True)
    asl.add_argument("--names")
    asl.add_argument("--source-padding", type=int, default=0)
    asl.set_defaults(func=cmd_atlas_slice)

    apk = a.add_parser("pack", parents=[common], help="Pack image files into an atlas + JSON")
    apk.add_argument("images", nargs="+")
    apk.add_argument("-o", "--out", required=True)
    apk.add_argument("--columns", type=int)
    apk.add_argument("--padding", type=int, default=0)
    apk.add_argument("--cell", help="Cell size N or WxH (default: first image's size)")
    apk.add_argument("--tileable", action="store_true", help="Wrap (not clamp) padding")
    apk.add_argument("--pot", action="store_true")
    apk.set_defaults(func=cmd_atlas_pack)

    agd = a.add_parser("guide", parents=[common], help="Draw a numbered grid layout image")
    agd.add_argument("--grid", required=True)
    agd.add_argument("--cell", default="256")
    agd.add_argument("--names")
    agd.add_argument("-o", "--out", required=True)
    agd.set_defaults(func=cmd_atlas_guide)
    # ---- uv (Blender UV atlas retexturing) ----
    uvp = sub.add_parser("uv", help="Retexture a Blender object's UV texture atlas").add_subparsers(dest="uv_command", required=True)

    ud = uvp.add_parser("demo", parents=[common], help="Make a demo .blend (UV-unwrapped house) + placeholder atlas + UV data")
    ud.add_argument("-o", "--out", required=True)
    ud.add_argument("--size", type=int, default=1024)
    ud.set_defaults(func=cmd_uv_demo)

    ue = uvp.add_parser("export", parents=[common], help="Export an object's UV islands/seams/textures from a .blend")
    ue.add_argument("blend")
    ue.add_argument("-o", "--out", required=True, help="Directory for uv_data.npz, uv_info.json, uv_preview.png")
    ue.add_argument("--object", help="Mesh object name (default: active / only UV-mapped mesh)")
    ue.add_argument("--uv-map", help="UV map name (default: active render map)")
    ue.add_argument("--size", help="Texture size N or WxH (default: the material's image size)")
    ue.set_defaults(func=cmd_uv_export)

    uph = uvp.add_parser("placeholder", parents=[common], help="Flat colour per island (a stand-in untextured atlas)")
    uph.add_argument("--uv", required=True)
    uph.add_argument("--base", help="Keep this image outside the islands")
    uph.add_argument("-o", "--out", required=True)
    uph.set_defaults(func=cmd_uv_placeholder)

    ur = uvp.add_parser("retexture", parents=[common], help="Repaint an atlas in the look of reference image(s), then check/repair")
    ur.add_argument("atlas", nargs="?", help="Current atlas (default: the texture recorded in uv_info.json)")
    ur.add_argument("--uv", required=True, help="UV export directory")
    ur.add_argument("--style", "-s", action="append", required=True, help="Reference image of the desired look (repeatable)")
    ur.add_argument("-o", "--out", required=True)
    ur.add_argument("--instruction", "-i", help="Extra direction, e.g. 'weathered, moss near the ground'")
    ur.add_argument("--material", "-M", action="append",
                    help='Material by surface, painted as its own masked pass: --material up="terracotta roof tiles" '
                         '--material side="fieldstone wall" (selectors: up, down, side, +x/-x/+y/-y/+z/-z, islands:1,2, all, rest)')
    ur.add_argument("--note", action="append", help='Per-island material: --note 3="terracotta roof tiles" (see uv_preview.png)')
    ur.add_argument("--judge", action="store_true",
                    help="Also run the vision-model review in the check (off by default: slow, and often wrong on rotated/mixed islands)")
    ur.add_argument("--padding", type=int, default=16, help="Bleed margin px (Blender bake default: 16)")
    ur.add_argument("--mode", default="auto", choices=["auto", "materials", "whole", "tiles", "islands"])
    ur.add_argument("--tile", type=int, default=1024)
    ur.add_argument("--repair-rounds", type=int, default=2, help="Repaint failing islands and re-check this many times")
    ur.add_argument("--no-fix-seams", action="store_true",
                    help="Don't blend tone across UV seams (by default only seams continuous in the original are blended)")
    ur.add_argument("--flatten-lighting", action="store_true", help="Remove large-scale baked light/shadow gradients")
    ur.set_defaults(func=cmd_uv_retexture)

    uc = uvp.add_parser("check", parents=[common], help="Check an atlas against the UV-atlas rules (exit 2 on failure)")
    uc.add_argument("image")
    uc.add_argument("--original", required=True, help="The atlas it replaces (same UV layout)")
    uc.add_argument("--uv", required=True)
    uc.add_argument("--raw", help="Unclamped model output (*.raw.png) for a stricter drift check")
    uc.add_argument("--padding", type=int, default=16)
    uc.add_argument("--style", help="Style reference (with --judge)")
    uc.add_argument("--judge", action="store_true", help="Add a vision-model review of the atlas (needs --style)")
    uc.add_argument("--material", "-M", action="append", help="The intended materials (tells the judge what goes where)")
    uc.add_argument("--instruction", "-i")
    uc.add_argument("--report", help="Also write the JSON report here")
    uc.set_defaults(func=cmd_uv_check)

    urp = uvp.add_parser("repair", parents=[common], help="Repaint specific islands (masked edits), then re-check")
    urp.add_argument("image")
    urp.add_argument("--original", required=True)
    urp.add_argument("--uv", required=True)
    urp.add_argument("--style", "-s", action="append", required=True)
    grp_r = urp.add_mutually_exclusive_group(required=True)
    grp_r.add_argument("--islands", help="Comma-separated island ids")
    grp_r.add_argument("--from-check", help="A check report JSON; repairs its repair_islands")
    urp.add_argument("--instruction", "-i")
    urp.add_argument("--note", action="append")
    urp.add_argument("--material", "-M", action="append", help="Same materials as the retexture run")
    urp.add_argument("--raw", help="The previous *.raw.png, so the re-check sees unclamped patches")
    urp.add_argument("--padding", type=int, default=16)
    urp.add_argument("-o", "--out", required=True)
    urp.set_defaults(func=cmd_uv_repair)

    ub = uvp.add_parser("bleed", parents=[common], help="Extend island colours into the padding margin")
    ub.add_argument("image")
    ub.add_argument("--uv", required=True)
    ub.add_argument("--padding", type=int, default=16)
    ub.add_argument("-o", "--out", required=True)
    ub.set_defaults(func=cmd_uv_bleed)

    ufs = uvp.add_parser("fix-seams", parents=[common], help="Blend tone across UV seams (then re-bleed)")
    ufs.add_argument("image")
    ufs.add_argument("--uv", required=True)
    ufs.add_argument("--width", type=int, default=6)
    ufs.add_argument("--original", help="Only blend seams that were continuous in this (original) atlas")
    ufs.add_argument("--padding", type=int, default=16)
    ufs.add_argument("-o", "--out", required=True)
    ufs.set_defaults(func=cmd_uv_fix_seams)

    ufh = uvp.add_parser("fill-holes", parents=[common], help="Fill thin unpainted strips inside islands (mirrors nearby texture)")
    ufh.add_argument("image")
    ufh.add_argument("--original", required=True, help="The atlas it replaces (to tell unpainted texels)")
    ufh.add_argument("--uv", required=True)
    ufh.add_argument("--material", "-M", action="append", help="Same materials as the retexture run")
    ufh.add_argument("--max-fill", type=float, default=0.05, help="Only fill islands with at most this fraction unpainted")
    ufh.add_argument("--raw", help="The *.raw.png: also fill strips uncovered by offset paint (recommended)")
    ufh.add_argument("--padding", type=int, default=16)
    ufh.add_argument("-o", "--out", required=True)
    ufh.set_defaults(func=cmd_uv_fill_holes)

    ufl = uvp.add_parser("flatten", parents=[common], help="Remove large-scale baked light/shadow gradients from an atlas")
    ufl.add_argument("image")
    ufl.add_argument("--original", required=True)
    ufl.add_argument("--uv", required=True)
    ufl.add_argument("--material", "-M", action="append", help="Flatten per material group (recommended)")
    ufl.add_argument("--padding", type=int, default=16)
    ufl.add_argument("-o", "--out", required=True)
    ufl.set_defaults(func=cmd_uv_flatten)

    urn = uvp.add_parser("render", parents=[common], help="Render the object in Blender (optionally with a new texture)")
    urn.add_argument("blend")
    urn.add_argument("-o", "--out", required=True, help="Directory for <view>.png")
    urn.add_argument("--image", help="Texture to apply (default: render the current one)")
    urn.add_argument("--object")
    urn.add_argument("--replace", help="Only swap Image Texture nodes using this file")
    urn.add_argument("--views", default="front,right,back,iso", help="front,back,left,right,top,bottom,iso,iso_back")
    urn.add_argument("--res", type=int, default=768)
    urn.add_argument("--engine", default="eevee", choices=["eevee", "workbench", "cycles"])
    urn.add_argument("--samples", type=int, default=32)
    urn.add_argument("--save", help="Also save a copy of the .blend with the new texture here")
    urn.add_argument("--opaque", action="store_true", help="Grey background instead of transparent PNGs")
    urn.set_defaults(func=cmd_uv_render)

    urr = uvp.add_parser("review-renders", parents=[common], help="Vision review of renders vs the style reference")
    urr.add_argument("--after", required=True, help="Directory of renders with the new texture")
    urr.add_argument("--before", help="Directory of renders with the old texture")
    urr.add_argument("--style", required=True)
    urr.set_defaults(func=cmd_uv_review_renders)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ImageGenError, FileNotFoundError, KeyError, ValueError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        if getattr(args, "json", False):
            print(json.dumps({"error": str(msg)}))
        print(f"error: {msg}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
