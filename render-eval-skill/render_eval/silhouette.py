"""Eval 4: silhouette overlap.

Compares the two foreground masks after framing alignment. Robust to lighting and
materials, and the best single signal for overall shape and proportions.

* ``iou`` and ``dice``: area overlap.
* ``boundary_f1``: outline agreement within the BSDS tolerance (0.75% of the diagonal).
* ``chamfer`` and ``hd95``: mean and 95th-percentile outline distance, as a
  fraction of the image side.
* ``area_ratio`` and ``aspect_ratio_error``: coarse proportion checks computed on
  the original (unaligned) foreground boxes.

score = IoU
"""

from __future__ import annotations

import numpy as np

from render_eval._geometry import boundary_prf, bsds_tolerance, mask_boundary
from render_eval.base import EvalConfig, EvalResult
from render_eval.pair import ImagePair

NAME = "silhouette"
DESCRIPTION = "Foreground mask IoU, Dice and outline distance"


def evaluate(pair: ImagePair, cfg: EvalConfig) -> EvalResult:
    a, b = pair.ref_mask, pair.ren_mask
    inter, union = int((a & b).sum()), int((a | b).sum())
    iou = inter / union if union else 1.0
    dice = 2 * inter / (int(a.sum()) + int(b.sum())) if union else 1.0
    bnd = boundary_prf(mask_boundary(b), mask_boundary(a), bsds_tolerance(pair.size))

    ref_aspect = pair.meta.get("reference", {}).get("bbox_aspect")
    ren_aspect = pair.meta.get("render", {}).get("bbox_aspect")
    aspect_err = abs(np.log(ren_aspect / ref_aspect)) if ref_aspect and ren_aspect else float("nan")

    metrics = {
        "iou": round(iou, 6),
        "dice": round(dice, 6),
        "boundary_f1": round(bnd["f1"], 6),
        "chamfer": round(bnd["chamfer_px"] / pair.size, 6),
        "hd95": round(bnd["hd95_px"] / pair.size, 6),
        "area_ratio": round(float(b.sum()) / max(1, int(a.sum())), 6),
        "aspect_ratio_error": round(float(aspect_err), 6),  # |log(render aspect / reference aspect)|
    }
    res = EvalResult(NAME, float(iou), metrics)
    if cfg.debug:
        vis = np.full(a.shape + (3,), 0.12, np.float32)
        vis[a & b] = (0.9, 0.9, 0.9)  # both
        vis[a & ~b] = (0.95, 0.3, 0.3)  # reference only (missing from render)
        vis[~a & b] = (0.3, 0.55, 0.95)  # render only (extra)
        res.artifacts["silhouette"] = vis
    return res
