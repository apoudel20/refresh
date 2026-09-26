"""Encode an eval run as vectors, so runs from different agent structures can be compared.

Two encodings per run:

* **Score vector** (8 numbers): each eval's score, in eval order
  ``pixel, depth, normals, silhouette, edges, embedding, color, judge``.
* **Token matrix** (8 tokens x 32 numbers = 256): one token per eval that summarises
  *what* the eval found, not just how well it scored. Every token has the same
  layout rules:

  - slot 0 is the eval's score;
  - the next slots are that eval's key metrics, scaled to about [0, 1] (or [-1, 1]
    when the sign matters, for example "render too large" vs "too small");
  - most tokens then carry a 4 x 4 grid over the aligned frame that says *where*
    the difference is (for depth: where the render is nearer or farther than the
    reference; for silhouette: where it has extra or missing area);
  - slot 31 is 1 when the eval ran and 0 when it was skipped.

  The embedding token carries a fixed random projection of the difference between
  the two image embeddings (which direction, semantically, the render is off).
  The judge token carries a projection of a text embedding of the critic's notes,
  so two runs with the same numbers but different complaints still differ.

Every slot has a stable name (``feature_names()``), so vectors from different runs
line up, and a distance between two runs can be broken down per eval.
Grid cells refer to the reference's aligned frame, which is the same for every run
against the same reference, so cells mean the same body region across runs.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from render_eval.base import EvalConfig
from render_eval.runner import EvalReport

VERSION = 1
TOKEN_DIM = 32
GRID = 4
ORDER = ("pixel", "depth", "normals", "silhouette", "edges", "embedding", "color", "judge")
EMBED_PROJ = 24
TEXT_PROJ = 16


def _grid_names(prefix: str) -> list[str]:
    return [f"{prefix}_r{i}c{j}" for i in range(GRID) for j in range(GRID)]


# Feature names per token, in slot order. Slot 31 is always "present".
SPEC: dict[str, list[str]] = {
    "pixel": ["score", "ssim_fg", "lpips_fg", "ssim", "lpips", "psnr_fg"] + _grid_names("lpips"),
    "depth": ["score", "spearman", "pearson", "ssi_mae", "aligned_nrmse", "discontinuity_f1", "overlap"]
    + _grid_names("nearer"),
    "normals": ["score", "mean_agreement", "median_agreement", "within_11", "within_30"] + _grid_names("angle_err"),
    "silhouette": ["score", "dice", "boundary_f1", "chamfer", "hd95", "area_log_ratio", "aspect_log_ratio"]
    + _grid_names("extra_minus_missing"),
    "edges": ["score", "precision", "recall", "interior_f1", "chamfer", "density_log_ratio"]
    + _grid_names("edge_surplus"),
    "embedding": ["score", "vectors_present"] + [f"diff_proj{k}" for k in range(EMBED_PROJ)],
    "color": ["score", "palette_emd", "palette_emd_ab", "delta_e2000", "d_lightness", "d_a", "d_b", "d_contrast"]
    + [f"share_{lvl}_{hue}" for hue in ("neutral", "warm", "cool") for lvl in ("dark", "mid_dark", "mid_light", "light")],
    "judge": ["score", "overall", "shape", "proportions", "parts", "color", "materials", "missing_parts",
              "extra_parts", "text_present"] + [f"text_proj{k}" for k in range(TEXT_PROJ)],
}
for _n, _names in SPEC.items():
    assert len(_names) <= TOKEN_DIM - 1, (_n, len(_names))


def feature_names() -> list[str]:
    """Names of all 256 slots of the flattened token matrix, e.g. ``depth.nearer_r1c2``."""
    out = []
    for n in ORDER:
        names = SPEC[n] + [f"pad{i}" for i in range(TOKEN_DIM - 1 - len(SPEC[n]))] + ["present"]
        out += [f"{n}.{x}" for x in names]
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _f(x: Any, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not math.isfinite(v) else float(min(hi, max(lo, v)))


def _grid(values: np.ndarray, mask: np.ndarray) -> list[float]:
    """Mean of ``values`` over ``mask`` in each cell of a GRID x GRID split; 0 where a cell has no mask pixels."""
    h, w = values.shape
    out = []
    for i in range(GRID):
        for j in range(GRID):
            ys, xs = slice(i * h // GRID, (i + 1) * h // GRID), slice(j * w // GRID, (j + 1) * w // GRID)
            m = mask[ys, xs]
            out.append(float(values[ys, xs][m].mean()) if m.any() else 0.0)
    return out


def _project(vec: Sequence[float], k: int, key: str, scale: float) -> list[float]:
    """Fixed random projection (seeded by ``key`` and input size), so the same input always maps the same way."""
    v = np.asarray(vec, np.float64)
    rng = np.random.default_rng(zlib.crc32(f"{key}:{v.size}:{k}".encode()))
    r = rng.standard_normal((v.size, k)) * (scale / math.sqrt(k))
    return (v @ r).tolist()


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, np.float64)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


# --------------------------------------------------------------------------- #
# Per-eval token builders: each returns {feature name: value}
# --------------------------------------------------------------------------- #


def _pixel(r, pair, cfg) -> dict[str, float]:
    m = r.metrics
    d = {"ssim_fg": _f(m["ssim_fg"]), "lpips_fg": _f(m["lpips_fg"], 0, 1), "ssim": _f(m["ssim"]),
         "lpips": _f(m["lpips"], 0, 1), "psnr_fg": _f(m["psnr_fg_db"] / 50.0, 0, 1)}
    lp = pair.cache.get("lpips_map")
    if lp is not None:
        d.update(zip(_grid_names("lpips"), _grid(lp, pair.union)))
    return d


def _depth(r, pair, cfg) -> dict[str, float]:
    from render_eval.depth import ssi_normalize

    m = r.metrics
    d = {"spearman": _f(m["spearman"]), "pearson": _f(m["pearson"]), "ssi_mae": _f(m["ssi_mae"] / 2.0, 0, 1),
         "aligned_nrmse": _f(m["aligned_nrmse"] / 1.5, 0, 1), "discontinuity_f1": _f(m["discontinuity_f1"], 0, 1),
         "overlap": _f(m["overlap_coverage"], 0, 1)}
    maps, region = pair.cache.get(("depth", cfg.depth_model)), pair.cache.get("depth_region")
    if maps is not None and region is not None and region.any():
        # Depth Anything predicts inverse depth: positive = render nearer the camera than the reference there.
        nearer = np.clip((ssi_normalize(maps[1], region) - ssi_normalize(maps[0], region)) / 2.0, -1, 1)
        d.update(zip(_grid_names("nearer"), _grid(nearer, region)))
    return d


def _normals(r, pair, cfg) -> dict[str, float]:
    m = r.metrics
    d = {"mean_agreement": _f(1 - m["mean_angle_deg"] / 90.0, 0, 1),
         "median_agreement": _f(1 - m["median_angle_deg"] / 90.0, 0, 1),
         "within_11": _f(m["within_11_25"], 0, 1), "within_30": _f(m["within_30"], 0, 1)}
    cached = pair.cache.get("normals_error")
    if cached is not None:
        err, region = cached
        d.update(zip(_grid_names("angle_err"), _grid(err / 90.0, region)))
    return d


def _silhouette(r, pair, cfg) -> dict[str, float]:
    m = r.metrics
    ref_a = pair.meta.get("reference", {}).get("bbox_aspect")
    ren_a = pair.meta.get("render", {}).get("bbox_aspect")
    d = {"dice": _f(m["dice"], 0, 1), "boundary_f1": _f(m["boundary_f1"], 0, 1),
         "chamfer": _f(m["chamfer"] * 10, 0, 1), "hd95": _f(m["hd95"] * 4, 0, 1),
         "area_log_ratio": _f(math.tanh(math.log(max(m["area_ratio"], 1e-6)))),
         "aspect_log_ratio": _f(math.tanh(math.log(ren_a / ref_a))) if ref_a and ren_a else 0.0}
    signed = pair.ren_mask.astype(np.float32) - pair.ref_mask.astype(np.float32)  # +1 extra, -1 missing
    d.update(zip(_grid_names("extra_minus_missing"), _grid(signed, np.ones_like(pair.ref_mask))))
    return d


def _edges(r, pair, cfg) -> dict[str, float]:
    m = r.metrics
    n_ref, n_ren = m["edge_pixels_reference"], m["edge_pixels_render"]
    d = {"precision": _f(m["precision"], 0, 1), "recall": _f(m["recall"], 0, 1),
         "interior_f1": _f(m["interior_f1"], 0, 1), "chamfer": _f(m["chamfer"] * 10, 0, 1),
         "density_log_ratio": _f(math.tanh(math.log((n_ren + 1) / (n_ref + 1))))}
    maps = pair.cache.get("edge_maps")
    if maps is not None:
        e_ref, e_ren = maps
        h, w = e_ref.shape
        cells = []
        for i in range(GRID):
            for j in range(GRID):
                ys, xs = slice(i * h // GRID, (i + 1) * h // GRID), slice(j * w // GRID, (j + 1) * w // GRID)
                a, b = int(e_ref[ys, xs].sum()), int(e_ren[ys, xs].sum())
                cells.append((b - a) / (a + b) if a + b else 0.0)  # +1 only render edges, -1 only reference edges
        d.update(zip(_grid_names("edge_surplus"), cells))
    return d


def _embedding(r, pair, cfg) -> dict[str, float]:
    ref_v, ren_v = r.details.get("reference_vector"), r.details.get("render_vector")
    if ref_v is None or ren_v is None:
        return {"vectors_present": 0.0}
    diff = _unit(ren_v) - _unit(ref_v)
    proj = _project(diff, EMBED_PROJ, f"embedding:{r.metrics.get('model')}", scale=2.0)
    return {"vectors_present": 1.0, **{f"diff_proj{k}": _f(v, -3, 3) for k, v in enumerate(proj)}}


def _color_bins(lab: np.ndarray) -> np.ndarray:
    """Share of pixels in 12 fixed bins: 4 lightness levels x (neutral, warm, cool)."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    level = np.clip((L // 25).astype(int), 0, 3)
    chroma, hue = np.hypot(a, b), np.degrees(np.arctan2(b, a))
    kind = np.where(chroma < 12, 0, np.where((hue > -45) & (hue < 135), 1, 2))  # warm = reds to yellows
    counts = np.bincount(kind * 4 + level, minlength=12).astype(np.float64)
    return counts / max(1, len(lab))


def _color(r, pair, cfg) -> dict[str, float]:
    from render_eval.color import foreground_lab

    m = r.metrics
    lab_ref, lab_ren = foreground_lab(pair.ref, pair.ref_mask), foreground_lab(pair.ren, pair.ren_mask)
    d = {"palette_emd": _f(m["palette_emd"] / 50.0, 0, 1), "palette_emd_ab": _f(m["palette_emd_ab"] / 25.0, 0, 1),
         "delta_e2000": _f(m["mean_delta_e2000"] / 50.0, 0, 1),
         "d_lightness": _f((lab_ren[:, 0].mean() - lab_ref[:, 0].mean()) / 50.0),
         "d_a": _f((lab_ren[:, 1].mean() - lab_ref[:, 1].mean()) / 30.0),
         "d_b": _f((lab_ren[:, 2].mean() - lab_ref[:, 2].mean()) / 30.0),
         "d_contrast": _f((lab_ren[:, 0].std() - lab_ref[:, 0].std()) / 30.0)}
    shares = _color_bins(lab_ren) - _color_bins(lab_ref)  # + = render has more of this color class
    names = [f"share_{lvl}_{hue}" for hue in ("neutral", "warm", "cool") for lvl in ("dark", "mid_dark", "mid_light", "light")]
    d.update(zip(names, shares.tolist()))
    return d


def critique_text(r) -> str:
    """The critic's notes as one string: per-criterion reasons, missing/extra parts and fixes."""
    d = r.details
    parts = [f"{c}: {d.get('reasons', {}).get(c, '')}" for c in ("shape", "proportions", "parts", "color", "materials")]
    parts.append("Missing: " + ", ".join(d.get("missing_parts") or []))
    parts.append("Extra: " + ", ".join(d.get("extra_parts") or []))
    parts.append("Fixes: " + " ".join(d.get("top_fixes") or []))
    return "\n".join(parts)


def _judge(r, pair, cfg, embedder: Any = None, notes: list[str] | None = None) -> dict[str, float]:
    m, dd = r.metrics, r.details
    d = {"overall": _f(m.get("overall", 0) / 10, 0, 1),
         **{c: _f(m.get(c, 0) / 10, 0, 1) for c in ("shape", "proportions", "parts", "color", "materials")},
         "missing_parts": _f(len(dd.get("missing_parts") or []) / 5, 0, 1),
         "extra_parts": _f(len(dd.get("extra_parts") or []) / 5, 0, 1), "text_present": 0.0}
    if embedder is False:
        return d
    try:
        if embedder is None:
            from render_eval.openrouter import OpenRouterEmbedder

            embedder = OpenRouterEmbedder(model=cfg.embedding_model)
        vec = embedder.embed_text(critique_text(r))
        proj = _project(_unit(vec), TEXT_PROJ, f"critique:{getattr(embedder, 'model', cfg.embedding_model)}", scale=1.0)
        d.update({"text_present": 1.0, **{f"text_proj{k}": _f(v, -3, 3) for k, v in enumerate(proj)}})
    except Exception as exc:  # no key, network error: keep the numeric part of the token
        if notes is not None:
            notes.append(f"critique text not embedded: {type(exc).__name__}: {exc}")
    return d


BUILDERS = {"pixel": _pixel, "depth": _depth, "normals": _normals, "silhouette": _silhouette,
            "edges": _edges, "embedding": _embedding, "color": _color}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


@dataclass
class RunVector:
    scores: list[float | None]  # 8, in ORDER; None where an eval did not run
    tokens: np.ndarray  # (8, 32)
    notes: list[str] = field(default_factory=list)

    def flat(self) -> np.ndarray:
        return self.tokens.reshape(-1)

    def score_array(self) -> np.ndarray:
        return np.array([0.0 if s is None else s for s in self.scores], np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "order": list(ORDER),
            "token_dim": TOKEN_DIM,
            "grid": GRID,
            "scores": [None if s is None else round(float(s), 6) for s in self.scores],
            "tokens": [[round(float(x), 5) for x in row] for row in self.tokens],
            **({"notes": self.notes} if self.notes else {}),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunVector":
        if d.get("version") != VERSION:
            raise ValueError(f"vector version {d.get('version')} does not match {VERSION}")
        return cls(scores=list(d["scores"]), tokens=np.asarray(d["tokens"], np.float64), notes=d.get("notes", []))


def encode(report: EvalReport, *, embedder: Any = None, embed_critique: bool = True) -> RunVector:
    """Build the score vector and token matrix for one evaluated pair.

    Needs the ``EvalReport`` from ``run_evals`` (its pair holds the maps the evals cached).
    Pass ``embed_critique=False`` to skip the one text-embedding call for the judge token.
    For the embedding token to carry its projection, run with ``EvalConfig(keep_vectors=True)``.
    """
    cfg: EvalConfig = report.config
    pair = report.pair
    tokens = np.zeros((len(ORDER), TOKEN_DIM), np.float64)
    scores: list[float | None] = []
    notes: list[str] = []
    for i, name in enumerate(ORDER):
        r = report.results.get(name)
        if r is None or not r.ok:
            scores.append(None)
            continue
        scores.append(float(r.score))
        if name == "judge":
            feats = _judge(r, pair, cfg, embedder if embed_critique else False, notes)
        else:
            feats = BUILDERS[name](r, pair, cfg)
        feats["score"] = float(r.score)
        for j, fname in enumerate(SPEC[name]):
            tokens[i, j] = _f(feats.get(fname, 0.0), -3, 3)
        tokens[i, TOKEN_DIM - 1] = 1.0
    return RunVector(scores=scores, tokens=tokens, notes=notes)


def token_distances(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Euclidean distance between two token matrices, per eval."""
    return {n: float(np.linalg.norm(a[i] - b[i])) for i, n in enumerate(ORDER)}


def distance_matrix(vectors: Iterable[np.ndarray]) -> np.ndarray:
    v = np.stack([np.asarray(x, np.float64).reshape(-1) for x in vectors])
    return np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)


def history_vectors(runs: list[dict[str, Any]], *, delta: bool = False) -> list[tuple[dict[str, Any], RunVector]]:
    """(run, vector) for every run in report.json that has a vector.

    With ``delta=True`` each vector becomes "this run minus the previous vectorised run on
    the same reference": what the run *changed*, which is often the fairer way to compare
    agent structures that started from different models.
    """
    out: list[tuple[dict[str, Any], RunVector]] = []
    last: dict[str, RunVector] = {}
    for run in runs:
        if "vector" not in run:
            continue
        try:
            v = RunVector.from_dict(run["vector"])
        except ValueError:
            continue
        key = run.get("reference_sha1", "")
        if delta:
            prev = last.get(key)
            last[key] = v
            if prev is None:
                continue
            diff_scores = [None if (s is None or p is None) else s - p for s, p in zip(v.scores, prev.scores)]
            v = RunVector(scores=diff_scores, tokens=v.tokens - prev.tokens)
        out.append((run, v))
    return out


def tokens_image(tokens: np.ndarray, cell: int = 22) -> np.ndarray:
    """Heatmap of the token matrix: one row per eval, blue negative, white zero, red positive."""
    from PIL import Image, ImageDraw

    from render_eval.report import _font

    label_w, top = 110, 26
    h, w = tokens.shape
    img = Image.new("RGB", (label_w + w * cell + 10, top + h * cell + 10), (250, 250, 248))
    d = ImageDraw.Draw(img)
    font = _font(13)
    d.text((label_w, 6), "token slots 0-31: score, metrics, 4x4 region grid ... present", fill=(95, 95, 90), font=font)
    blue, red, white = np.array([45, 100, 210]), np.array([210, 60, 45]), np.array([255, 255, 255])
    for i in range(h):
        d.text((8, top + i * cell + 4), ORDER[i], fill=(27, 27, 25), font=font)
        for j in range(w):
            t = float(np.clip(tokens[i, j], -1, 1))
            c = white + (red - white) * t if t >= 0 else white + (blue - white) * (-t)
            x0, y0 = label_w + j * cell, top + i * cell
            d.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=tuple(int(v) for v in c))
    return np.asarray(img, np.float32) / 255.0
