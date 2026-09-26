"""Eval 2: depth matching.

Runs a monocular depth estimator (Depth Anything V2) on both images and compares
the two predicted depth maps. Monocular depth is only relative (unknown scale and
shift), so every metric here is scale-and-shift invariant:

* ``spearman`` / ``pearson``: rank / linear correlation of the two depth maps.
* ``ssi_mae``: mean absolute difference after normalising each map by its median
  and mean absolute deviation (the MiDaS "scale-shift invariant" loss).
* ``aligned_nrmse``: RMSE after least-squares fitting the render's depth to the
  reference's, divided by the reference's standard deviation.
* ``discontinuity_f1``: agreement of depth edges (where parts overlap or occlude).

Metrics are computed where both objects are present (mask intersection), so this
eval measures interior shape; outline mismatch is the silhouette eval's job.

score = max(0, spearman)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from render_eval._geometry import auto_canny, boundary_prf, bsds_tolerance, colorize, hstack
from render_eval.base import EvalConfig, EvalResult, clip01
from render_eval.pair import ImagePair

NAME = "depth"
DESCRIPTION = "Depth Anything V2 depth maps compared with scale/shift-invariant metrics"


def predict_depth(pair: ImagePair, cfg: EvalConfig) -> tuple[np.ndarray, np.ndarray]:
    """Relative inverse depth (larger = closer) for (reference, render), cached on the pair."""
    key = ("depth", cfg.depth_model)
    if key in pair.cache:
        return pair.cache[key]

    import torch
    import torch.nn.functional as F

    from render_eval.models import depth_model, pick_device

    device = pick_device(cfg.device)
    processor, model = depth_model(cfg.depth_model, device)
    inputs = processor(images=[pair.pil("ref"), pair.pil("ren")], return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth  # (2, h, w)
    pred = F.interpolate(pred[:, None].float(), size=(pair.size, pair.size), mode="bicubic", align_corners=False)
    out = pred[:, 0].cpu().numpy().astype(np.float32)
    pair.cache[key] = (out[0], out[1])
    return pair.cache[key]


def ssi_normalize(d: np.ndarray, region: np.ndarray) -> np.ndarray:
    """(d - median) / mean(|d - median|) using statistics from ``region`` only."""
    vals = d[region]
    t = float(np.median(vals))
    s = float(np.mean(np.abs(vals - t)))
    return (d - t) / (s if s > 1e-8 else 1.0)


def compare_maps(ref_d: np.ndarray, ren_d: np.ndarray, region: np.ndarray) -> dict[str, float]:
    a, b = ref_d[region].astype(np.float64), ren_d[region].astype(np.float64)
    if a.size < 16 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return {"spearman": 0.0, "pearson": 0.0, "ssi_mae": float("nan"), "aligned_nrmse": float("nan")}
    rng = np.random.default_rng(0)
    idx = rng.choice(a.size, size=min(a.size, 60_000), replace=False)
    spearman = float(spearmanr(a[idx], b[idx]).statistic)
    pearson = float(pearsonr(a, b).statistic)
    ssi_mae = float(np.mean(np.abs(ssi_normalize(ref_d, region)[region] - ssi_normalize(ren_d, region)[region])))
    A = np.stack([b, np.ones_like(b)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, a, rcond=None)
    nrmse = float(np.sqrt(np.mean((s * b + t - a) ** 2)) / np.std(a))
    return {"spearman": spearman, "pearson": pearson, "ssi_mae": ssi_mae, "aligned_nrmse": nrmse}


def evaluate(pair: ImagePair, cfg: EvalConfig) -> EvalResult:
    ref_d, ren_d = predict_depth(pair, cfg)
    inter, union = pair.intersection, pair.union
    coverage = float(inter.sum() / max(1, union.sum()))
    region, region_name = (inter, "intersection") if inter.sum() >= 0.01 * pair.size**2 else (union, "union")

    pair.cache["depth_region"] = region  # reused by render_eval.vectorize
    m = compare_maps(ref_d, ren_d, region)
    tol = bsds_tolerance(pair.size)
    band = union  # depth edges only matter on or around the objects
    e_ref = auto_canny(ssi_normalize(ref_d, union), band)
    e_ren = auto_canny(ssi_normalize(ren_d, union), band)
    disc = boundary_prf(e_ren, e_ref, tol)

    metrics = {k: round(v, 6) for k, v in m.items()}
    metrics.update(
        {
            "discontinuity_f1": round(disc["f1"], 6),
            "region": region_name,
            "overlap_coverage": round(coverage, 6),
            "model": cfg.depth_model,
        }
    )
    res = EvalResult(NAME, clip01(max(0.0, m["spearman"])), metrics)
    if cfg.debug:
        diff = np.abs(ssi_normalize(ref_d, region) - ssi_normalize(ren_d, region))
        res.artifacts["depth"] = hstack(colorize(ref_d, union), colorize(ren_d, union), colorize(diff, region))
    return res
