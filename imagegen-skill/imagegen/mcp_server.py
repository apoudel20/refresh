"""MCP server exposing the imagegen pipeline as tools, for Codex / Claude Code / any MCP client.

Run: ``uv run imagegen-mcp`` (stdio). Every tool writes files and returns a JSON summary plus,
where it helps, a small preview image so the calling agent can see what it made and iterate.
Paths may be absolute or relative to the server's working directory (the project root).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Image as MCPImage
from mcp.server.mcpserver import MCPServer
from PIL import Image

from imagegen import atlas, blender_bridge, ops, pipeline, uvtex
from imagegen.backends import backend_status, get_backend

server = MCPServer(
    "imagegen",
    instructions=(
        "Image generation/editing pipeline (texture atlases, sprites, seamless textures). "
        "Backends: 'codex' uses the user's ChatGPT/Codex subscription via the codex CLI (default); "
        "'openrouter' uses OPENROUTER_API_KEY. Each model call takes ~30-90s. "
        "Atlas workflow: atlas_init_spec -> edit spec -> atlas_build -> atlas_review(fix=true) -> "
        "atlas_edit_cell for targeted fixes (region edits only change pixels inside the region) -> "
        "atlas_revert_cell if an edit made things worse. "
        "Blender UV atlases (retexture an object's texture from a style reference): uv_export -> "
        "uv_retexture (use materials like ['up=roof tiles','side=stone wall']) -> read its check report -> "
        "uv_render + uv_review_renders to verify on the actual mesh -> uv_repair failing islands."
    ),
)


def _preview(img: Image.Image, max_side: int = 512) -> MCPImage:
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return MCPImage(data=buf.getvalue(), format="png")


def _reply(summary: dict[str, Any], image: Image.Image | None = None, preview: bool = True) -> list[Any]:
    out: list[Any] = [json.dumps(summary, indent=2, default=str)]
    if preview and image is not None:
        out.append(_preview(image))
    return out


def _region(region: str | None, size: tuple[int, int]):
    return ops.parse_region(region, size) if region else None


# --------------------------------------------------------------------------- #
# single images
# --------------------------------------------------------------------------- #


@server.tool(structured_output=False)
def generate_image(
    prompt: str,
    out: str,
    references: list[str] | None = None,
    size: str | None = None,
    aspect_ratio: str | None = None,
    transparent_key: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    preview: bool = True,
) -> list[Any]:
    """Generate an image from a text prompt (optionally guided by reference image paths) and save it to `out`.

    size: exact output size "N" or "WxH" (generated then cropped/resized). transparent_key: e.g. "#ff00ff" —
    ask for that flat background in the prompt and it is keyed out to alpha.
    """
    images, res = pipeline.generate(
        prompt, backend=get_backend(backend, model), references=references or [],
        size=ops.parse_size(size) if size else None, aspect_ratio=aspect_ratio,
    )  # fmt: skip
    img = images[0]
    if transparent_key:
        img = ops.chroma_key(img, ops.parse_color(transparent_key))
    path = ops.save_image(img, out)
    return _reply({"output": str(path), "size": list(img.size), "backend": res.backend, "model": res.model}, img, preview)


@server.tool(structured_output=False)
def edit_image(
    image: str,
    instruction: str,
    out: str | None = None,
    region: str | None = None,
    mask: str | None = None,
    context: float = 0.5,
    references: list[str] | None = None,
    backend: str | None = None,
    model: str | None = None,
    preview: bool = True,
) -> list[Any]:
    """Edit an image with a natural-language instruction.

    Limit the change to `region` ("x,y,w,h" in pixels, or fractions like "0.5,0,0.5,0.5") or to a `mask` image
    (white = editable): pixels outside it are guaranteed unchanged. Writes to `out` (default: overwrite `image`).
    """
    base = ops.load_image(image)
    res = pipeline.edit(base, instruction, backend=get_backend(backend, model), region=_region(region, base.size),
                        mask=mask, context=context, references=references or [])  # fmt: skip
    path = ops.save_image(res.image, out or image)
    return _reply({"output": str(path), **res.info()}, res.image, preview)


@server.tool(structured_output=False)
def upscale_image(
    image: str,
    out: str,
    scale: float = 2.0,
    method: str = "lanczos",
    hint: str | None = None,
    backend: str | None = None,
    model: str | None = None,
) -> list[Any]:
    """Enlarge an image. method: lanczos | bicubic | nearest (local, free; nearest for pixel art) |
    ai (model re-renders overlapping tiles with more detail; layout can't drift)."""
    b = get_backend(backend, model) if method == "ai" else None
    img = pipeline.upscale(image, scale, method=method, backend=b, hint=hint or "")
    path = ops.save_image(img, out)
    return _reply({"output": str(path), "size": list(img.size), "method": method}, img)


@server.tool(structured_output=False)
def make_seamless(
    image: str,
    out: str,
    method: str = "blend",
    band: float = 0.2,
    backend: str | None = None,
    model: str | None = None,
) -> list[Any]:
    """Make a texture tile seamlessly. method: blend (local cross-fade, free) | ai (model repaints the seams).
    Returns seam scores (~1.0 = invisible seam, >1.6 = visible) and a 2x2 tiled preview."""
    before = ops.seam_score(ops.load_image(image))
    b = get_backend(backend, model) if method == "ai" else None
    img = pipeline.seamless(image, method=method, backend=b, band=band)
    path = ops.save_image(img, out)
    return _reply(
        {"output": str(path), "seam_score_before": round(before, 3), "seam_score_after": round(ops.seam_score(img), 3)},
        ops.tile_preview(img, 2),
    )


@server.tool(structured_output=False)
def chroma_key(image: str, out: str, color: str = "#ff00ff", tolerance: float = 60, softness: float = 40) -> list[Any]:
    """Make a flat background colour transparent (for sprites generated on a solid key colour)."""
    img = ops.chroma_key(ops.load_image(image), ops.parse_color(color), tolerance, softness)
    return _reply({"output": str(ops.save_image(img, out))}, img)


@server.tool()
def imagegen_status() -> dict[str, Any]:
    """Which backends are usable (codex install/login, OpenRouter key) and their default models."""
    return backend_status()


# --------------------------------------------------------------------------- #
# atlases
# --------------------------------------------------------------------------- #


@server.tool()
def atlas_init_spec(path: str, example: str = "textures") -> dict[str, Any]:
    """Write an example atlas spec JSON ("textures": tileable terrain; "sprites": transparent item icons).

    Spec fields: name, cell_size, columns, padding, style, tileable, background (opaque|transparent),
    key_color, mode (cells: one call per cell, most accurate | sheet: whole grid in one call, most consistent),
    resample (lanczos|nearest), power_of_two, style_reference, chain_style, auto_seamless,
    cells: [{name, prompt, tileable?, reference?}].
    """
    data = atlas.EXAMPLE_SPRITES if example == "sprites" else atlas.EXAMPLE_SPEC
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")
    return {"spec": str(p), "content": data}


@server.tool(structured_output=False)
def atlas_build(
    spec: str,
    out_dir: str,
    only: list[str] | None = None,
    force: bool = False,
    mode: str | None = None,
    concurrency: int = 4,
    backend: str | None = None,
    model: str | None = None,
) -> list[Any]:
    """Generate an atlas project from a spec JSON path: cells/<name>.png, atlas.png, atlas.json (frames + UVs).

    Resumable: existing cells are kept unless force=true. `only` limits to named cells.
    """
    s = atlas.AtlasSpec.load(spec)
    if mode:
        s.mode = mode
    res = atlas.build(s, out_dir, get_backend(backend, model), only=only, force=force, concurrency=concurrency)
    return _reply(res, ops.load_image(res["atlas"]))


@server.tool(structured_output=False)
def atlas_edit_cell(
    project_dir: str,
    name: str,
    instruction: str | None = None,
    prompt: str | None = None,
    image: str | None = None,
    region: str | None = None,
    references: list[str] | None = None,
    backend: str | None = None,
    model: str | None = None,
) -> list[Any]:
    """Change one atlas cell and repack. Give exactly one of:
    instruction (edit current cell; optional `region` "x,y,w,h" in cell pixels limits the change),
    prompt (new description, regenerates the cell), image (replace with this file).
    The previous version is kept in history/ (see atlas_revert_cell)."""
    proj = atlas.AtlasProject(project_dir)
    spec = proj.spec()
    b = get_backend(backend, model) if (instruction or prompt) else None
    res = atlas.edit_cell(project_dir, name, b, instruction=instruction, prompt=prompt, image=image,
                          region=_region(region, spec.cell_size), references=references or [])  # fmt: skip
    return _reply(res, proj.load_cell(name))


@server.tool()
def atlas_revert_cell(project_dir: str, name: str, steps: int = 1) -> dict[str, Any]:
    """Restore a cell to a previous version from its history and repack."""
    return atlas.revert_cell(project_dir, name, steps)


@server.tool()
def atlas_review(
    project_dir: str,
    fix: bool = False,
    threshold: int = 7,
    rounds: int = 2,
    consistency: bool = True,
    only: list[str] | None = None,
    backend: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Vision-model QA: score every cell 0-10 against its description (plus seam checks for tileable
    cells and a whole-atlas style-consistency check). fix=true edits/regenerates failing cells and re-reviews."""
    return atlas.review(project_dir, get_backend(backend, model), fix=fix, threshold=threshold, rounds=rounds,
                        consistency=consistency, only=only)  # fmt: skip


@server.tool()
def atlas_repack(project_dir: str, padding: int | None = None, columns: int | None = None) -> dict[str, Any]:
    """Rebuild atlas.png/atlas.json from cells/ (optionally changing padding or column count)."""
    proj = atlas.AtlasProject(project_dir)
    if padding is not None or columns is not None:
        spec = proj.spec()
        spec.padding = spec.padding if padding is None else padding
        spec.columns = spec.columns if columns is None else columns
        spec.save(proj.spec_path)
    return atlas.repack(project_dir)


@server.tool()
def atlas_upscale(
    project_dir: str, scale: float = 2.0, method: str = "lanczos", backend: str | None = None, model: str | None = None
) -> dict[str, Any]:
    """Upscale every cell (grid stays exact), update cell_size, repack."""
    b = get_backend(backend, model) if method == "ai" else None
    return atlas.upscale_atlas(project_dir, scale, method=method, backend=b)


@server.tool()
def atlas_import(
    image: str,
    grid: str,
    out_dir: str,
    names: list[str] | None = None,
    prompts: list[str] | None = None,
    source_padding: int = 0,
    style: str = "",
    tileable: bool = False,
) -> dict[str, Any]:
    """Turn an existing atlas image ("COLSxROWS" grid) into an editable project; add prompts so review works."""
    return atlas.import_atlas(image, ops.parse_grid(grid), out_dir, names=names, prompts=prompts,
                              source_padding=source_padding, style=style, tileable=tileable)  # fmt: skip


@server.tool()
def atlas_pack(images: list[str], out: str, columns: int | None = None, padding: int = 0, tileable: bool = False) -> dict[str, Any]:
    """Pack image files into one atlas PNG plus a JSON of frames/UVs (named by file stem)."""
    return atlas.pack_images(images, out, columns=columns, padding=padding, tileable=tileable)


@server.tool()
def atlas_slice(image: str, grid: str, out_dir: str, names: list[str] | None = None, source_padding: int = 0) -> dict[str, Any]:
    """Cut a grid image ("COLSxROWS") into separate cell PNGs."""
    paths = atlas.slice_image(image, ops.parse_grid(grid), out_dir, names=names, source_padding=source_padding)
    return {"cells": [str(p) for p in paths]}



# --------------------------------------------------------------------------- #
# Blender UV atlases
# --------------------------------------------------------------------------- #


@server.tool(structured_output=False)
def uv_export(blend: str, out_dir: str, object: str | None = None, uv_map: str | None = None, size: str | None = None) -> list[Any]:
    """Export a mesh object's UV layout from a .blend: island map, per-texel normals and UV rotation,
    seams, texel density, the image textures on its material, and UV sanity counts. Returns a summary
    and a numbered island preview (use the numbers for island selectors/notes)."""
    blender_bridge.export_uv(blend, out_dir, obj=object, uv_map=uv_map, size=size)
    uv = uvtex.UVLayout.load(out_dir)
    prev = uv.preview()
    prev.save(Path(out_dir) / "uv_preview.png")
    info = uv.info
    summary = {"object": info["object"], "uv_map": info["uv_map"], "size": info["size"], "textures": info["textures"],
               "sanity": info["sanity"], "seams": len(info["seams"]),
               "islands": [{k: i.get(k) for k in ("id", "bbox", "texels", "normal", "up_rotation")} for i in info["islands"]]}  # fmt: skip
    return _reply(summary, prev)


@server.tool(structured_output=False)
def uv_retexture(
    uv_dir: str,
    styles: list[str],
    out: str,
    atlas: str | None = None,
    materials: list[str] | None = None,
    instruction: str = "",
    padding: int = 16,
    mode: str = "auto",
    repair_rounds: int = 2,
    judge: bool = False,
    backend: str | None = None,
    model: str | None = None,
) -> list[Any]:
    """Repaint a Blender object's UV texture atlas in the look of style reference image(s), clamp the
    result to the UV islands, bleed the padding, then check it against the atlas rules and repair
    failing islands. materials: ["up=terracotta roof tiles", "side=fieldstone wall", "down=dark slab"]
    (selectors up/down/side/+x/-x/+y/-y/+z/-z/islands:1,2/all/rest) paints each surface type in its own
    upright pass — strongly recommended. Writes out, out.raw.png, out.guide.png, out.check.json.
    The check can miss things: always verify with uv_render + uv_review_renders."""
    uv = uvtex.UVLayout.load(uv_dir)
    atlas_path = atlas or next((t["filepath"] for t in uv.info.get("textures", []) if t.get("filepath")), None)
    if not atlas_path:
        raise ValueError("No atlas given and none recorded in uv_info.json")
    b = get_backend(backend, model)
    mats = uvtex.parse_materials(materials)
    res = uvtex.retexture(atlas_path, uv, styles, backend=b, instruction=instruction, materials=mats,
                          padding=padding, mode=mode)  # fmt: skip
    final, raw = res.image, res.raw
    rep: dict[str, Any] = {}
    for rnd in range(repair_rounds + 1):
        final, _ = uvtex.finish(final, atlas_path, uv, materials=mats, padding=padding, raw=raw)
        rep = uvtex.check(final, atlas_path, uv, raw=raw, padding=padding, style=styles[0] if judge else None,
                          backend=b if judge else None, materials=mats, instruction=instruction)  # fmt: skip
        if rep["passed"] or not rep["repair_islands"] or rnd == repair_rounds:
            break
        final, raw = uvtex.repair(final, atlas_path, uv, rep["repair_islands"], styles, backend=b, raw=raw,
                                  instruction=instruction, materials=mats, padding=padding)  # fmt: skip
    p = ops.save_image(final, out)
    stem = Path(out)
    ops.save_image(raw, stem.with_name(f"{stem.stem}.raw{stem.suffix or '.png'}"))
    ops.save_image(res.guide, stem.with_name(f"{stem.stem}.guide{stem.suffix or '.png'}"))
    stem.with_name(f"{stem.stem}.check.json").write_text(json.dumps(rep, indent=2, default=str))
    summary = {"output": str(p), "passed": rep["passed"], "errors": rep["errors"], "warnings": rep["warnings"],
               "repair_islands": rep["repair_islands"], "checks": [{k: c[k] for k in ("name", "passed", "severity", "detail")} for c in rep["checks"]],
               "groups": res.groups}  # fmt: skip
    return _reply(summary, uvtex.outline(final, uv))


@server.tool()
def uv_check(image: str, original: str, uv_dir: str, raw: str | None = None, padding: int = 16,
             style: str | None = None, judge: bool = False, materials: list[str] | None = None,
             backend: str | None = None, model: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Check a (re)textured atlas against the UV-atlas rules: size/format, island coverage, padding
    bleed, drift vs UV outlines (pass the .raw.png), seam tone continuity, untouched empty space, UV
    sanity, and optionally a vision review (judge=true + style). passed=false lists repair_islands."""
    b = get_backend(backend, model) if judge else None
    return uvtex.check(image, original, uvtex.UVLayout.load(uv_dir), raw=raw, padding=padding, style=style,
                       backend=b, materials=uvtex.parse_materials(materials))  # fmt: skip


@server.tool(structured_output=False)
def uv_repair(image: str, original: str, uv_dir: str, islands: list[int], styles: list[str], out: str,
              raw: str | None = None, materials: list[str] | None = None, instruction: str = "",
              padding: int = 16, backend: str | None = None, model: str | None = None) -> list[Any]:  # fmt: skip
    """Repaint specific UV islands (masked, upright edits; pixels elsewhere untouched), then re-check."""
    uv = uvtex.UVLayout.load(uv_dir)
    final, new_raw = uvtex.repair(image, original, uv, islands, styles, backend=get_backend(backend, model), raw=raw,
                                  instruction=instruction, materials=uvtex.parse_materials(materials), padding=padding)  # fmt: skip
    mats = uvtex.parse_materials(materials)
    final, _ = uvtex.finish(final, original, uv, materials=mats, padding=padding, raw=new_raw)
    p = ops.save_image(final, out)
    rep = uvtex.check(final, original, uv, raw=new_raw, padding=padding, materials=mats)
    return _reply({"output": str(p), "repaired": islands, "passed": rep["passed"], "errors": rep["errors"],
                   "warnings": rep["warnings"]}, uvtex.outline(final, uv))  # fmt: skip


@server.tool(structured_output=False)
def uv_render(blend: str, out_dir: str, image: str | None = None, object: str | None = None,
              views: str = "front,right,back,iso", res: int = 512, engine: str = "eevee") -> list[Any]:  # fmt: skip
    """Render the object in Blender (background; the .blend is never modified), optionally with a new
    texture swapped in. Returns the render paths and a contact sheet."""
    out = blender_bridge.render(blend, out_dir, image=image, obj=object, views=views, res=res, engine=engine)
    ims = [ops.load_image(p).convert("RGBA") for p in out["renders"]]
    sheet, _ = ops.pack_grid(ims, min(4, len(ims)), ims[0].size, padding=2)
    bg = Image.new("RGB", sheet.size, (128, 128, 128))
    bg.paste(sheet, (0, 0), sheet)
    return _reply(out, bg)


@server.tool()
def uv_review_renders(after_dir: str, style: str, before_dir: str | None = None,
                      backend: str | None = None, model: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Vision review of Blender renders against the style reference: style match, visible seams,
    stretching/misalignment, baked lighting, pattern scale/direction — with fix suggestions."""
    after = sorted(Path(after_dir).glob("*.png"))
    before = sorted(Path(before_dir).glob("*.png")) if before_dir else []
    return uvtex.review_renders(after, style, backend=get_backend(backend, model), before=before)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
