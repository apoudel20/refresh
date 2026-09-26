"""Step-by-step report: one reference against several candidates, through every eval.

For each candidate the report shows what each step produced (masks, depth maps,
normal maps, edge maps, palettes, embedding vectors, the judge's reasoning), a
one-line verdict, the score and the raw metrics. It writes three files:

* ``report.html``: self-contained page (images embedded), open it in a browser.
* ``summary.png``: one overview image, a row per candidate.
* ``report.json``: every number, for scripts or MongoDB.

    render-eval report test-assets/dog-1.png test-assets/*.png -o outputs/dog-1-report
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_eval._geometry import hstack, mask_boundary
from render_eval.base import EvalConfig, EvalResult
from render_eval.pair import ImageInput, ImagePair, to_uint8
from render_eval.runner import EVALS, EvalReport, jsonable, run_evals

TITLES = {
    "pixel": "Pixel similarity",
    "depth": "Depth",
    "normals": "Surface normals",
    "silhouette": "Silhouette",
    "edges": "Edges",
    "embedding": "Embedding (OpenRouter)",
    "color": "Color palette",
    "judge": "VLM judge (OpenRouter)",
}
SHORT = {"pixel": "pixel", "depth": "depth", "normals": "normals", "silhouette": "silh", "edges": "edges",
         "embedding": "embed", "color": "color", "judge": "judge"}


def captions(cfg: EvalConfig) -> dict[str, str]:
    normals_src = "Marigold" if cfg.normals_backend == "marigold" else "Depth-derived"
    return {
        "alignment": "Both images cropped to the object and put on grey. The red line is the object mask every step uses.",
        "pixel": "Reference | candidate | absolute color error | LPIPS perceptual distance (bright = more different).",
        "depth": "Depth Anything V2 output: reference | candidate (bright = closer) | difference after normalising scale.",
        "normals": f"{normals_src} normals: reference | candidate (color = surface direction) | angle error (bright = larger).",
        "silhouette": "White = both objects. Red = only in the reference (missing from the candidate). Blue = only in the candidate (extra).",
        "edges": "Green = reference edges. Magenta = candidate edges. White = exact overlap.",
        "color": "One shared 8-color palette. Bar widths show each image's area share of each color: reference (top), candidate (bottom).",
        "embedding": "Raw model output: the first 256 vector dimensions, reference (top) and candidate (bottom). Blue = negative, red = positive.",
    }


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


def alignment_panel(pair: ImagePair) -> np.ndarray:
    def outline(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = img.copy()
        edge = cv2.dilate(mask_boundary(mask).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        out[edge] = (1.0, 0.2, 0.2)
        return out

    return hstack(outline(pair.ref, pair.ref_mask), outline(pair.ren, pair.ren_mask))


def vector_strip(a: list[float], b: list[float], dims: int = 256, cell: int = 4, row_h: int = 28) -> np.ndarray:
    va, vb = np.asarray(a[:dims], np.float32), np.asarray(b[:dims], np.float32)
    scale = max(float(np.abs(np.concatenate([va, vb])).max()), 1e-8)

    def row(v: np.ndarray) -> np.ndarray:
        t = np.clip(v / scale, -1, 1)[None, :, None]  # -1 blue, 0 white, +1 red
        blue, red, white = np.array([0.2, 0.4, 0.85]), np.array([0.85, 0.25, 0.2]), np.ones(3)
        rgb = np.where(t < 0, white + (blue - white) * -t, white + (red - white) * t)
        return np.repeat(np.repeat(rgb, row_h, axis=0), cell, axis=1)

    gap = np.ones((6, len(va) * cell, 3))
    return np.concatenate([row(va), gap, row(vb)], axis=0).astype(np.float32)


def result_panel(r: EvalResult) -> np.ndarray | None:
    if r.name in r.artifacts:
        return r.artifacts[r.name]
    if r.name == "embedding" and "reference_vector" in r.details:
        return vector_strip(r.details["reference_vector"], r.details["render_vector"])
    return None


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #


def _band(x: float, cuts: tuple[float, float, float], words: tuple[str, str, str, str]) -> str:
    return words[0] if x >= cuts[0] else words[1] if x >= cuts[1] else words[2] if x >= cuts[2] else words[3]


def explain(r: EvalResult) -> str:
    """One plain-English line saying what this step concluded, derived only from its metrics."""
    if r.skipped:
        return f"Skipped: {r.skipped}"
    if r.error:
        return f"Failed: {r.error}"
    m, s = r.metrics, r.score or 0.0
    if r.name == "pixel":
        word = _band(s, (0.8, 0.45, 0.25), ("Nearly the same pixels", "Similar image with visible pixel differences",
                                            "Pixels differ a lot", "Unrelated pixels"))
        return f"{word}. On the object, LPIPS is {m['lpips_fg']:.2f} (0 = identical) and SSIM is {m['ssim_fg']:.2f} (1 = identical)."
    if r.name == "depth":
        word = _band(m["spearman"], (0.85, 0.6, 0.3), ("Same depth layout", "Similar depth layout",
                                                       "Weakly related depth", "Different depth layout"))
        return (f"{word}. Relative depth rank correlation is {m['spearman']:.2f} over the "
                f"{m['overlap_coverage']:.0%} of the frame both objects share.")
    if r.name == "normals":
        word = _band(m["within_22_5"], (0.75, 0.5, 0.3), ("Surfaces face the same way almost everywhere",
                                                          "Most surfaces agree", "Many surfaces disagree",
                                                          "Surface orientation mostly disagrees"))
        return f"{word}. {m['within_22_5']:.0%} of shared surface is within 22.5 degrees; mean error {m['mean_angle_deg']:.0f} degrees."
    if r.name == "silhouette":
        word = _band(m["iou"], (0.9, 0.75, 0.62), ("Outlines match", "Similar outline", "Rough outline match",
                                                   "Different outline (about 0.5 is typical for unrelated shapes)"))
        return f"{word}. IoU {m['iou']:.2f}, outline F1 {m['boundary_f1']:.2f}, mean outline gap {m['chamfer']:.1%} of the frame."
    if r.name == "edges":
        word = _band(m["f1"], (0.8, 0.5, 0.3), ("Edges line up", "Many edges line up", "Few edges line up",
                                                "Edges do not line up"))
        return (f"{word}. {m['recall']:.0%} of the reference's edges are reproduced, and {m['precision']:.0%} of the "
                f"candidate's edges exist in the reference.")
    if r.name == "embedding":
        word = _band(m["cosine"], (0.9, 0.7, 0.45), ("Same subject and look", "Same kind of subject",
                                                     "Loosely related", "Unrelated"))
        return f"{word}. Cosine similarity {m['cosine']:.2f} between {m['dimensions']}-dimension {m['model']} vectors."
    if r.name == "color":
        d = m["palette_emd"]
        word = "Same colors" if d < 3 else "Similar colors" if d < 8 else "Noticeably different colors" if d < 20 else "Different colors"
        return (f"{word}. Palette distance is Delta E {d:.1f}, or {m['palette_emd_ab']:.1f} ignoring lightness "
                f"(about 2 is invisible, 10 is obvious).")
    if r.name == "judge":
        crit = ", ".join(f"{c} {m[c]}" for c in ("shape", "proportions", "parts", "color", "materials"))
        return f"Overall {m['overall']}/10 from {m['model']}. Criteria: {crit}."
    return ""


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1b1b19;--muted:#5c5c57;--line:#e3e3de;--accent:#0f766e;--imgbg:#e9e9e4}
@media (prefers-color-scheme: dark){:root{--bg:#111110;--panel:#1b1b19;--ink:#f2f2ef;--muted:#a3a39c;--line:#2e2e2a;--accent:#2dd4bf;--imgbg:#2a2a27}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:21px;margin:48px 0 8px;padding-top:20px;border-top:1px solid var(--line)}
h2 .score{float:right}
.meta{color:var(--muted);font-size:13px}
.lead{max-width:760px}
.scroll{overflow-x:auto}
table.summary{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px;margin-top:12px}
.summary th,.summary td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
.summary th{font-weight:600;color:var(--muted);font-size:12.5px}
.summary th:first-child,.summary td:first-child{text-align:left}
.summary td.c{font-weight:700}
.summary a{color:inherit}
.summary img{width:40px;height:40px;border-radius:4px;vertical-align:middle;margin-right:8px}
ol.steps{padding-left:20px;color:var(--muted);font-size:14px}
ol.steps b{color:var(--ink)}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}
.row figure{margin:0;flex:1 1 220px;min-width:0}
figure img,.card img{max-width:100%;height:auto;border-radius:6px;background:var(--imgbg);display:block}
figcaption,.caption{color:var(--muted);font-size:12.5px;margin-top:6px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.card h3{margin:0;font-size:16px;display:flex;justify-content:space-between;align-items:center;gap:12px}
.score{font-variant-numeric:tabular-nums;font-weight:700}
.bar{height:6px;width:110px;background:var(--line);border-radius:3px;overflow:hidden;display:inline-block;vertical-align:middle;margin-left:10px}
.bar>i{display:block;height:100%;background:var(--accent)}
.what{color:var(--muted);font-size:13px;margin:2px 0 8px}
.verdict{margin:0 0 10px}
details{margin-top:10px;font-size:13px}
summary{cursor:pointer;color:var(--muted)}
dl.m{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:2px 16px;margin:8px 0 0}
dl.m div{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dotted var(--line)}
dl.m dt{color:var(--muted)} dl.m dd{margin:0;font-variant-numeric:tabular-nums}
.chips{margin:4px 0}
.chips span{display:inline-flex;align-items:center;gap:6px;margin:0 12px 6px 0;font-size:12.5px;font-variant-numeric:tabular-nums}
.chips b{width:14px;height:14px;border-radius:3px;display:inline-block;border:1px solid var(--line)}
.crit{margin:6px 0 0;padding-left:18px}
.crit li{margin:3px 0}
.warn{color:#b45309}
"""


def _img_uri(arr: np.ndarray, max_w: int | None = None, fmt: str = "JPEG") -> str:
    img = Image.fromarray(to_uint8(arr)) if arr.dtype != np.uint8 else Image.fromarray(arr)
    if max_w and img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=88)
    else:
        img.save(buf, format="PNG", optimize=True)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _original(src: ImageInput) -> np.ndarray:
    if isinstance(src, np.ndarray):
        return src
    img = src if isinstance(src, Image.Image) else Image.open(src)
    rgba = np.asarray(img.convert("RGBA"), np.float32) / 255.0
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + 0.85 * (1 - a)  # show transparency on light grey


def _e(x: Any) -> str:
    return html.escape(str(x))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return _e(v)


def _card(r: EvalResult, idx: int, caption: str | None) -> str:
    mod = EVALS[r.name]
    score = f"{r.score:.3f}" if r.ok else "n/a"
    bar = f'<span class="bar"><i style="width:{(r.score or 0) * 100:.1f}%"></i></span>' if r.ok else ""
    parts = [
        f'<div class="card"><h3><span>{idx} · {_e(TITLES[r.name])}</span>'
        f'<span><span class="score">{score}</span>{bar}</span></h3>',
        f'<div class="what">{_e(mod.DESCRIPTION)} · {r.seconds:.1f}s</div>',
        f'<p class="verdict">{_e(explain(r))}</p>',
    ]
    panel = result_panel(r)
    if panel is not None:
        flat = r.name in ("silhouette", "edges", "color", "embedding")
        parts.append(f'<img alt="{_e(r.name)} output" src="{_img_uri(panel, 1100, "PNG" if flat else "JPEG")}">')
        if caption:
            parts.append(f'<div class="caption">{_e(caption)}</div>')

    if r.name == "color" and r.ok:
        for label, key in (("Reference", "reference_palette"), ("Candidate", "render_palette")):
            chips = "".join(f'<span><b style="background:{c["hex"]}"></b>{c["hex"]} {c["share"]:.0%}</span>'
                            for c in r.details.get(key, []))
            parts.append(f'<div class="chips"><span style="min-width:74px">{label}</span>{chips}</div>')
    if r.name == "judge" and r.ok:
        reasons = r.details.get("reasons", {})
        items = "".join(f"<li><b>{_e(c)} {r.metrics.get(c)}/10</b>: {_e(reasons.get(c, ''))}</li>"
                        for c in ("shape", "proportions", "parts", "color", "materials"))
        parts.append(f'<ul class="crit">{items}</ul>')
        for label, key in (("Missing parts", "missing_parts"), ("Extra parts", "extra_parts")):
            vals = r.details.get(key) or []
            parts.append(f"<p><b>{label}:</b> {_e(', '.join(vals)) if vals else 'none'}</p>")
        fixes = r.details.get("top_fixes") or []
        if fixes:
            parts.append("<p><b>Top fixes:</b></p><ol class=\"crit\">" + "".join(f"<li>{_e(f)}</li>" for f in fixes) + "</ol>")

    if r.metrics:
        rows = "".join(f"<div><dt>{_e(k)}</dt><dd>{_fmt(v)}</dd></div>" for k, v in r.metrics.items())
        parts.append(f'<details><summary>Raw metrics</summary><dl class="m">{rows}</dl></details>')
    if r.error and "traceback" in r.details:
        parts.append(f"<details><summary>Traceback</summary><pre>{_e(r.details['traceback'])}</pre></details>")
    parts.append("</div>")
    return "".join(parts)


def render_html(reference: ImageInput, runs: dict[str, tuple[ImageInput, EvalReport]], cfg: EvalConfig) -> str:
    names = list(next(iter(runs.values()))[1].results) if runs else []
    ref_name = Path(reference).name if isinstance(reference, (str, Path)) else "reference"
    caps = captions(cfg)
    when = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    head = "".join(f"<th>{_e(SHORT[n])}</th>" for n in names)
    rows = []
    for cname, (_, rep) in runs.items():
        cells = "".join(
            f'<td style="background:color-mix(in srgb,var(--accent) {(r.score or 0) * 38:.0f}%,transparent)">'
            f'{r.score:.2f}</td>' if r.ok else "<td>n/a</td>"
            for r in rep.results.values()
        )
        comp = f"{rep.composite:.2f}" if rep.composite is not None else "n/a"
        thumb = _img_uri(rep.pair.ren, 80)
        rows.append(f'<tr><td><a href="#{_e(cname)}"><img alt="" src="{thumb}">{_e(cname)}</a></td>'
                    f'<td class="c">{comp}</td>{cells}</tr>')

    steps = "".join(f"<li><b>{_e(TITLES[n])}</b>: {_e(EVALS[n].DESCRIPTION)}</li>" for n in names)
    out = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>Eval report: {_e(ref_name)}</title><style>{CSS}</style></head><body><main>",
        f"<h1>Reference vs candidates: step-by-step eval report</h1>",
        f'<p class="meta">Reference <b>{_e(ref_name)}</b> · {len(runs)} candidates · {when} · '
        f"judge {_e(cfg.judge_model)} · embeddings {_e(cfg.embedding_model)} · normals {_e(cfg.normals_backend)}</p>",
        '<p class="lead">Every candidate goes through the same pipeline. Step 0 finds the object in each image and aligns '
        "the two. Steps 1 to 8 each compare the aligned pair and score it from 0 to 1, where higher means a closer match. "
        "The composite is a weighted mean of the steps.</p>",
        f'<div class="row"><figure style="max-width:420px"><img alt="reference" src="{_img_uri(_original(reference), 840)}">'
        f"<figcaption>Reference: {_e(ref_name)}</figcaption></figure>",
        f'<div style="flex:1 1 320px"><ol class="steps" start="0"><li><b>Preprocess</b>: find the object '
        "(alpha channel or background-removal model), crop both images to it, put both on grey</li>"
        f"{steps}</ol></div></div>",
        f'<div class="scroll"><table class="summary"><thead><tr><th>Candidate</th><th>composite</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>',
    ]

    for cname, (src, rep) in runs.items():
        pair, meta = rep.pair, rep.pair.meta
        comp = f"{rep.composite:.3f}" if rep.composite is not None else "n/a"
        out.append(f'<h2 id="{_e(cname)}">{_e(ref_name)} vs {_e(cname)}<span class="score">composite {comp}</span></h2>')

        def mask_note(m: dict[str, Any]) -> str:
            src_word = {"alpha": "the alpha channel", "segmentation": f"the background-removal model ({cfg.mask_model})",
                        "provided": "a mask file", "none": "nothing (no object found, whole image used)"}[m["mask_source"]]
            return f"mask from {src_word}, object covers {m['foreground_fraction']:.0%} of the original"

        warn = "".join(f'<p class="warn">Warning: {_e(w)}</p>' for w in meta.get("warnings", []))
        out.append(
            '<div class="card"><h3><span>0 · Preprocess</span><span class="score">'
            f"{rep.seconds:.1f}s total</span></h3>"
            f'<div class="what">Reference: {_e(mask_note(meta["reference"]))}. Candidate: {_e(mask_note(meta["render"]))}.</div>'
            f'<div class="row"><figure><img alt="reference original" src="{_img_uri(_original(reference), 560)}">'
            f"<figcaption>Reference original</figcaption></figure>"
            f'<figure><img alt="candidate original" src="{_img_uri(_original(src), 560)}">'
            f"<figcaption>Candidate original</figcaption></figure></div>"
            f'<img style="margin-top:12px" alt="aligned pair" src="{_img_uri(alignment_panel(pair), 1100)}">'
            f'<div class="caption">{_e(caps["alignment"])}</div>{warn}</div>'
        )
        for n, r in rep.results.items():
            out.append(_card(r, list(EVALS).index(n) + 1, caps.get(n)))
    out.append("</main></body></html>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Overview PNG
# --------------------------------------------------------------------------- #


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/Library/Fonts/Arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_summary_png(runs: dict[str, tuple[ImageInput, EvalReport]], tile: int = 190) -> Image.Image:
    columns = [("Aligned: reference | candidate", "alignment"), ("Silhouette", "silhouette"),
               ("Depth: ref | cand | diff", "depth"), ("Normals: ref | cand | error", "normals"), ("Edges", "edges")]
    label_w, pad, head_h = 330, 14, 44
    bg, ink, muted = (250, 250, 248), (27, 27, 25), (100, 100, 95)

    rows = []
    for cname, (_, rep) in runs.items():
        panels = {"alignment": alignment_panel(rep.pair)}
        for n, r in rep.results.items():
            if n in r.artifacts:
                panels[n] = r.artifacts[n]
        imgs = []
        for _, key in columns:
            p = panels.get(key)
            if p is None:
                imgs.append(None)
                continue
            im = Image.fromarray(to_uint8(p))
            imgs.append(im.resize((round(im.width * tile / im.height), tile), Image.LANCZOS))
        rows.append((cname, rep, imgs))

    col_w = [max((r[2][i].width if r[2][i] else tile) for r in rows) for i in range(len(columns))]
    width = label_w + sum(col_w) + pad * (len(columns) + 1)
    height = head_h + len(rows) * (tile + pad) + pad
    sheet = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(sheet)
    f_head, f_name, f_comp, f_small = _font(16, True), _font(20, True), _font(18, True), _font(15)

    x = label_w + pad
    for (title, _), w in zip(columns, col_w):
        d.text((x, 14), title, fill=muted, font=f_head)
        x += w + pad
    for i, (cname, rep, imgs) in enumerate(rows):
        y = head_h + i * (tile + pad)
        d.text((pad, y + 4), cname, fill=ink, font=f_name)
        comp = f"{rep.composite:.2f}" if rep.composite is not None else "n/a"
        d.text((pad, y + 32), f"composite {comp}", fill=(15, 118, 110), font=f_comp)
        scores = [(SHORT[n], r.score) for n, r in rep.results.items()]
        for j, (n, s) in enumerate(scores):
            cx, cy = pad + (j % 2) * 150, y + 64 + (j // 2) * 24
            d.text((cx, cy), f"{n:<7} {s:.2f}" if s is not None else f"{n:<7} n/a", fill=ink, font=f_small)
        x = label_w + pad
        for im, w in zip(imgs, col_w):
            if im is not None:
                sheet.paste(im, (x, y))
            x += w + pad
    return sheet


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _unique_names(candidates: list[ImageInput]) -> list[str]:
    names, seen = [], {}
    for i, c in enumerate(candidates):
        base = Path(c).stem if isinstance(c, (str, Path)) else f"candidate-{i + 1}"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return names


def build_report(
    reference: ImageInput,
    candidates: list[ImageInput],
    out_dir: str | Path,
    cfg: EvalConfig | None = None,
    *,
    evals: list[str] | None = None,
    weights: dict[str, float] | None = None,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
) -> dict[str, Any]:
    """Run every candidate against ``reference`` and write report.html, summary.png and report.json."""
    cfg = replace(cfg or EvalConfig(), debug=True, keep_vectors=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs: dict[str, tuple[ImageInput, EvalReport]] = {}
    for name, cand in zip(_unique_names(candidates), candidates):
        log(f"[{len(runs) + 1}/{len(candidates)}] {name} ...")
        rep = run_evals(reference, cand, evals, cfg, weights=weights)
        runs[name] = (cand, rep)
        comp = f"{rep.composite:.3f}" if rep.composite is not None else "n/a"
        log(f"    composite {comp}  ({rep.seconds:.1f}s)")

    (out / "report.html").write_text(render_html(reference, runs, cfg))
    render_summary_png(runs).save(out / "summary.png")
    data = {
        "reference": str(reference) if isinstance(reference, (str, Path)) else None,
        "candidates": {n: {"source": str(src) if isinstance(src, (str, Path)) else None, **rep.to_dict()}
                       for n, (src, rep) in runs.items()},
    }
    for c in data["candidates"].values():  # vectors are big and live in MongoDB, not in the report file
        emb = c["results"].get("embedding", {}).get("details", {})
        emb.pop("reference_vector", None)
        emb.pop("render_vector", None)
    (out / "report.json").write_text(json.dumps(jsonable(data), indent=2))
    return {"runs": runs, "files": [out / "report.html", out / "summary.png", out / "report.json"]}


def summary_table(runs: dict[str, tuple[ImageInput, EvalReport]]) -> str:
    names = list(next(iter(runs.values()))[1].results) if runs else []
    width = max([len(n) for n in runs] + [9])
    lines = [f"{'candidate':<{width}}  {'comp':>5}  " + "  ".join(f"{SHORT[n]:>6}" for n in names)]
    for cname, (_, rep) in runs.items():
        comp = f"{rep.composite:.2f}" if rep.composite is not None else "  n/a"
        vals = "  ".join(f"{r.score:>6.2f}" if r.ok else f"{'n/a':>6}" for r in rep.results.values())
        lines.append(f"{cname:<{width}}  {comp:>5}  {vals}")
    return "\n".join(lines)
