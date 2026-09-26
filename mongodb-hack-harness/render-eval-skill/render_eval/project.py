"""Project mode: record every run, keep the latest run's images, print the critique.

This is what ``render-eval critique`` (and so the render-eval skill) runs. For one reference/render pair it:

1. appends the run (scores, metrics, critic output) to ``<reports>/report.json``,
   which holds every run recorded in the project;
2. deletes ``<reports>/latest-run/`` and rewrites it with this run's comparison
   images (aligned pair, pixel, depth, normals, silhouette, edges, embedding,
   color), an ``overview.png`` with all of them, ``report.html`` and ``critique.md``;
3. returns the critique as a string: composite score, change since the previous
   run on the same reference, the critic's notes and top fixes, and each step's verdict.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageDraw

from render_eval.base import EvalConfig
from render_eval.pair import ImageInput, to_uint8
from render_eval.report import (
    TITLES,
    _font,
    _original,
    alignment_panel,
    explain,
    render_html,
    result_panel,
)
from render_eval.runner import EVALS, EvalReport, jsonable, run_evals
from render_eval.vectorize import ORDER, RunVector, encode, feature_names, token_distances, tokens_image

REPORT_FILE = "report.json"
LATEST_DIR = "latest-run"
MARKER = ".render-eval-latest-run"
OLD_MARKERS = (".harness-latest-run",)  # written by the earlier in-repo version
STEP_FILES = {
    "alignment": "00-aligned.png",
    "pixel": "01-pixel.png",
    "depth": "02-depth.png",
    "normals": "03-normals.png",
    "silhouette": "04-silhouette.png",
    "edges": "05-edges.png",
    "embedding": "06-embedding.png",
    "color": "07-color.png",
}
CRITERIA = ("shape", "proportions", "parts", "color", "materials")


# --------------------------------------------------------------------------- #
# History file
# --------------------------------------------------------------------------- #


def _sha1_and_path(src: ImageInput) -> tuple[str | None, str]:
    if isinstance(src, (str, Path)):
        p = Path(src).expanduser().resolve()
        return str(p), hashlib.sha1(p.read_bytes()).hexdigest()
    arr = np.asarray(src)
    return None, hashlib.sha1(arr.tobytes()).hexdigest()


def load_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": 1, "runs": []}
    data = json.loads(path.read_text())
    data.setdefault("runs", [])
    return data


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


@contextmanager
def _locked(reports: Path) -> Iterator[None]:
    """Serialise writers so two runs finishing together cannot drop each other's history entry."""
    try:
        import fcntl
    except ImportError:  # Windows: no advisory locks; accept the small race
        yield
        return
    with open(reports / ".report.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _entry(report: EvalReport, reference: ImageInput, candidate: ImageInput, label: str | None, now: dt.datetime) -> dict[str, Any]:
    ref_path, ref_sha = _sha1_and_path(reference)
    cand_path, cand_sha = _sha1_and_path(candidate)
    stem = Path(cand_path).stem if cand_path else "candidate"
    data = report.to_dict()
    emb = data["results"].get("embedding", {}).get("details", {})
    emb.pop("reference_vector", None)  # 1024 floats each; not worth keeping in the history file
    emb.pop("render_vector", None)
    return {
        "id": f"{now:%Y%m%dT%H%M%SZ}-{stem}",
        "timestamp": now.isoformat(timespec="seconds"),
        "label": label,
        "reference": ref_path,
        "candidate": cand_path,
        "reference_sha1": ref_sha,
        "candidate_sha1": cand_sha,
        **data,
    }


# --------------------------------------------------------------------------- #
# Critique text
# --------------------------------------------------------------------------- #


def _rel(p: Path) -> str:
    try:
        return os.path.relpath(p)
    except ValueError:
        return str(p)


def _name(path: str | None) -> str:
    return Path(path).name if path else "<image>"


def format_critique(
    entry: dict[str, Any],
    report: EvalReport,
    previous: dict[str, Any] | None,
    best: dict[str, Any] | None,
    runs_for_reference: int,
    latest_dir: Path,
    history_path: Path,
    total_runs: int,
    vec: RunVector | None = None,
    nearest: tuple[dict[str, Any], float, dict[str, float]] | None = None,
) -> str:
    comp = entry.get("composite")
    comp_s = f"{comp:.3f}" if comp is not None else "n/a"
    lines = [
        f"# Render critique: {_name(entry['candidate'])} vs {_name(entry['reference'])}",
        "",
        f"Run {entry['id']}" + (f" ({entry['label']})" if entry.get("label") else "")
        + f". Run {runs_for_reference} for this reference.",
        f"Composite score: {comp_s} out of 1.000 (higher is a closer match).",
    ]

    if previous and previous.get("composite") is not None and comp is not None:
        delta = comp - previous["composite"]
        changes = []
        for k, s in entry["scores"].items():
            p = previous.get("scores", {}).get(k)
            if s is not None and p is not None and abs(s - p) >= 0.005:
                changes.append((abs(s - p), f"{k} {s - p:+.2f}"))
        top = ", ".join(c for _, c in sorted(changes, reverse=True)[:4]) or "no step moved by more than 0.005"
        prev_label = f", {previous['label']}" if previous.get("label") else ""
        lines.append(
            f"Since the previous run ({_name(previous.get('candidate'))}{prev_label}): "
            f"{previous['composite']:.3f} to {comp:.3f} ({delta:+.3f}). Biggest changes: {top}."
        )
    else:
        lines.append("This is the first recorded run for this reference.")
    if best and best.get("composite") is not None and comp is not None:
        if best["id"] == entry["id"]:
            lines.append("This is the best run so far for this reference.")
        else:
            lines.append(f"Best so far for this reference: {best['composite']:.3f} ({best['id']}).")

    judge = report.results.get("judge")
    lines.append("")
    if judge is None:
        lines.append("## Critic: not run (the judge eval was not selected)")
    elif not judge.ok:
        lines.append(f"## Critic: unavailable ({judge.skipped or judge.error})")
    else:
        m, d = judge.metrics, judge.details
        lines.append(f"## Critic ({m.get('model')}): overall {m.get('overall')}/10")
        for c in CRITERIA:
            lines.append(f"- {c} {m.get(c)}/10: {d.get('reasons', {}).get(c, '')}")
        lines.append(f"Missing parts: {', '.join(d.get('missing_parts') or []) or 'none'}")
        lines.append(f"Extra parts: {', '.join(d.get('extra_parts') or []) or 'none'}")
        fixes = d.get("top_fixes") or []
        if fixes:
            lines.append("Top fixes, most important first:")
            lines.extend(f"{i}. {f}" for i, f in enumerate(fixes, 1))

    lines += ["", "## Step scores"]
    for n, r in report.results.items():
        score = f"{r.score:.2f}" if r.ok else "n/a"
        lines.append(f"- {list(EVALS).index(n) + 1}. {TITLES[n]} {score}: {explain(r)}")

    if vec is not None:
        vals = ", ".join("n/a" if s is None else f"{s:.2f}" for s in vec.scores)
        lines += [
            "",
            "## Vectors",
            f"- Score vector ({', '.join(ORDER)}): [{vals}]",
            f"- Token matrix: {vec.tokens.shape[0]} tokens x {vec.tokens.shape[1]} numbers, one token per step, "
            f"saved in {_rel(latest_dir / 'vector.json')} and in the history file.",
        ]
        if nearest is not None:
            run, dist, per = nearest
            top = ", ".join(f"{k} {v:.2f}" for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:3])
            lab = f" ({run['label']})" if run.get("label") else ""
            lines.append(f"- Most similar earlier run by tokens: {run['id']}{lab}, distance {dist:.2f}. "
                         f"Tokens that differ most: {top}.")
        lines.extend(f"- Note: {n}" for n in vec.notes)

    lines += [
        "",
        "## Files",
        f"- {_rel(latest_dir / 'overview.png')}: every step's output in one image",
        f"- {_rel(latest_dir)}/: the individual step images, report.html and critique.md for this run only",
        f"- {_rel(history_path)}: " + (f"all {total_runs} recorded runs" if total_runs != 1 else "1 recorded run")
        + " with full metrics",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Latest-run folder
# --------------------------------------------------------------------------- #


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: Any, width: int) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=font) <= width or not line:
                line = trial
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def overview_image(report: EvalReport, entry: dict[str, Any], width: int = 1200, vec: RunVector | None = None) -> Image.Image:
    """All step outputs stacked in one image, each with its title, score and verdict."""
    pad, inner = 20, width - 40
    bg, ink, muted, accent = (250, 250, 248), (27, 27, 25), (95, 95, 90), (15, 118, 110)
    f_title, f_head, f_body = _font(26, True), _font(19, True), _font(15)
    canvas = Image.new("RGB", (width, 20000), bg)
    d = ImageDraw.Draw(canvas)
    y = pad

    def text_block(text: str, font: Any, fill: tuple[int, int, int], gap: int = 4) -> None:
        nonlocal y
        for line in _wrap(d, text, font, inner):
            d.text((pad, y), line, fill=fill, font=font)
            y += font.size + gap

    comp = entry.get("composite")
    text_block(f"{_name(entry['candidate'])} vs {_name(entry['reference'])}", f_title, ink)
    text_block(f"Composite {comp:.3f}" if comp is not None else "Composite n/a", f_head, accent)
    text_block(f"Run {entry['id']}" + (f" ({entry['label']})" if entry.get("label") else ""), f_body, muted)
    y += 10

    steps: list[tuple[str, str | None, np.ndarray | None, str]] = [
        ("0 · Preprocess: aligned reference | candidate", None, alignment_panel(report.pair),
         "Both images cropped to the object and put on grey; the red line is the object mask.")
    ]
    for n, r in report.results.items():
        steps.append((f"{list(EVALS).index(n) + 1} · {TITLES[n]}", f"{r.score:.2f}" if r.ok else "n/a",
                      result_panel(r), explain(r)))

    for title, score, panel, verdict in steps:
        d.line([(pad, y), (width - pad, y)], fill=(225, 225, 220), width=1)
        y += 12
        d.text((pad, y), title, fill=ink, font=f_head)
        if score is not None:
            d.text((width - pad - d.textlength(score, font=f_head), y), score, fill=accent, font=f_head)
        y += f_head.size + 8
        text_block(verdict, f_body, muted)
        y += 6
        if panel is not None:
            im = Image.fromarray(to_uint8(panel))
            flat = im.width / im.height > 4  # thin strips (embedding vector, palettes): short and crisp
            scale = min(inner / im.width, (110 if flat else 380) / im.height)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                           Image.NEAREST if flat else Image.LANCZOS)
            canvas.paste(im, (pad, y))
            y += im.height + 14

    judge = report.results.get("judge")
    if judge is not None and judge.ok:
        dd = judge.details
        text_block("Critic notes", f_head, ink)
        for c in CRITERIA:
            text_block(f"{c} {judge.metrics.get(c)}/10: {dd.get('reasons', {}).get(c, '')}", f_body, ink)
        for i, fix in enumerate(dd.get("top_fixes") or [], 1):
            text_block(f"Fix {i}: {fix}", f_body, accent)
    if vec is not None:
        y += 10
        d.line([(pad, y), (width - pad, y)], fill=(225, 225, 220), width=1)
        y += 12
        text_block("Token matrix: one 32-number token per step (blue negative, red positive)", f_head, ink)
        im = Image.fromarray(to_uint8(tokens_image(vec.tokens)))
        canvas.paste(im, (pad, y))
        y += im.height + 10
    return canvas.crop((0, 0, width, y + pad))


def _save_original(src: ImageInput, path: Path) -> None:
    if isinstance(src, (str, Path)):
        Image.open(src).save(path)
    else:
        Image.fromarray(to_uint8(_original(src))).save(path)


def write_latest_run(
    latest: Path, report: EvalReport, reference: ImageInput, candidate: ImageInput, entry: dict[str, Any],
    critique: str, cfg: EvalConfig, vec: RunVector | None = None,
) -> list[Path]:
    """Replace ``latest`` with this run's images and text. Refuses to delete a folder it did not create."""
    if latest.exists():
        ours = any((latest / m).exists() for m in (MARKER, *OLD_MARKERS))
        if not ours and any(latest.iterdir()):
            raise RuntimeError(f"{latest} exists but was not created by render-eval; not deleting it")
        shutil.rmtree(latest)
    latest.mkdir(parents=True)
    (latest / MARKER).write_text("Created by render-eval. This folder is deleted and rewritten on every run.\n")

    written: list[Path] = []
    _save_original(reference, latest / "reference.png")
    _save_original(candidate, latest / "candidate.png")
    panels = {"alignment": alignment_panel(report.pair)}
    for n, r in report.results.items():
        p = result_panel(r)
        if p is not None:
            panels[n] = p
    for n, arr in panels.items():
        path = latest / STEP_FILES[n]
        Image.fromarray(to_uint8(arr)).save(path)
        written.append(path)

    overview_image(report, entry, vec=vec).save(latest / "overview.png")
    if vec is not None:
        Image.fromarray(to_uint8(tokens_image(vec.tokens))).save(latest / "08-tokens.png")
        (latest / "vector.json").write_text(json.dumps({**vec.to_dict(), "feature_names": feature_names()}, indent=1))
        written += [latest / "08-tokens.png", latest / "vector.json"]
    name = Path(entry["candidate"]).stem if entry.get("candidate") else "candidate"
    (latest / "report.html").write_text(render_html(reference, {name: (candidate, report)}, cfg))
    (latest / "critique.md").write_text(critique)
    (latest / "run.json").write_text(json.dumps(jsonable(entry), indent=2))
    written += [latest / f for f in ("reference.png", "candidate.png", "overview.png", "report.html", "critique.md", "run.json")]
    return written


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def record_run(
    reference: ImageInput,
    candidate: ImageInput,
    reports_dir: str | Path = "reports",
    cfg: EvalConfig | None = None,
    *,
    evals: list[str] | None = None,
    weights: dict[str, float] | None = None,
    label: str | None = None,
    meta: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
    embed_critique: bool = True,
) -> dict[str, Any]:
    """Evaluate, append to report.json, rewrite latest-run/, and return the critique string.

    ``meta`` is stored with the run as-is, e.g. ``{"structure": "geometry+fur+materials"}``, so runs
    can be grouped by the agent structure that produced them.
    """
    cfg = replace(cfg or EvalConfig(), debug=True, keep_vectors=True)
    report = run_evals(reference, candidate, evals, cfg, weights=weights)
    vec = encode(report, embed_critique=embed_critique)

    reports = Path(reports_dir).expanduser().resolve()
    reports.mkdir(parents=True, exist_ok=True)
    history_path, latest = reports / REPORT_FILE, reports / LATEST_DIR
    now = now or dt.datetime.now(dt.timezone.utc)

    with _locked(reports):
        history = load_history(history_path)
        entry = _entry(report, reference, candidate, label, now)
        entry["meta"] = meta or {}
        entry["vector"] = vec.to_dict()
        same_ref = [r for r in history["runs"] if r.get("reference_sha1") == entry["reference_sha1"]]
        nearest = None
        for r in same_ref:
            if "vector" not in r:
                continue
            try:
                other = RunVector.from_dict(r["vector"])
            except ValueError:
                continue
            dist = float(np.linalg.norm(other.tokens - vec.tokens))
            if nearest is None or dist < nearest[1]:
                nearest = (r, dist, token_distances(vec.tokens, other.tokens))
        previous = same_ref[-1] if same_ref else None
        scored = [r for r in same_ref + [entry] if r.get("composite") is not None]
        best = max(scored, key=lambda r: r["composite"]) if scored else None
        critique = format_critique(
            entry, report, previous, best, len(same_ref) + 1, latest, history_path, len(history["runs"]) + 1,
            vec=vec, nearest=nearest,
        )
        entry["critique"] = critique
        write_latest_run(latest, report, reference, candidate, entry, critique, cfg, vec)
        history["runs"].append(jsonable(entry))
        history["updated"] = entry["timestamp"]
        _atomic_write(history_path, json.dumps(history, indent=2))

    return {"entry": entry, "critique": critique, "report": report, "vector": vec, "latest_dir": latest,
            "history_path": history_path}
