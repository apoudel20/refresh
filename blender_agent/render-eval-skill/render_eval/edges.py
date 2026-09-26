"""Eval 5: edge and contour agreement.

Runs Canny on both aligned images and matches the edge maps with a pixel
tolerance, the way boundary-detection benchmarks do. Catches internal structure
that a silhouette misses: part boundaries, creases, eye sockets, panel lines.

A light blur is applied first so fine texture (fur, noise) does not dominate, and
Canny thresholds adapt to each image's contrast. Edges are only counted on or
near the objects.

* ``f1`` / ``precision`` / ``recall``: all edges, BSDS tolerance.
  Precision is "render edges that exist in the reference", recall is
  "reference edges reproduced by the render".
* ``interior_f1``: same, ignoring a band around both outlines, so it scores
  internal structure only.
* ``chamfer``: mean edge-to-edge distance as a fraction of the image side.

score = f1
"""

from __future__ import annotations

import cv2
import numpy as np

from render_eval._geometry import auto_canny, boundary_prf, bsds_tolerance
from render_eval.base import EvalConfig, EvalResult
from render_eval.pair import ImagePair

NAME = "edges"
DESCRIPTION = "Canny edge maps matched with a pixel tolerance (F-score, Chamfer)"


def edge_map(img: np.ndarray, region: np.ndarray, blur_sigma: float) -> np.ndarray:
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    return auto_canny(gray, region)


def evaluate(pair: ImagePair, cfg: EvalConfig) -> EvalResult:
    tol = bsds_tolerance(pair.size)
    blur = pair.size / 512.0 * 1.5
    k = int(2 * round(tol) + 1)
    region = cv2.dilate(pair.union.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    e_ref = edge_map(pair.ref, region, blur)
    e_ren = edge_map(pair.ren, region, blur)
    pair.cache["edge_maps"] = (e_ref, e_ren)  # reused by render_eval.vectorize
    allm = boundary_prf(e_ren, e_ref, tol)

    # Interior: drop everything within ~2 tolerances of either outline.
    k2 = int(4 * round(tol) + 1)
    interior = cv2.erode(pair.intersection.astype(np.uint8), np.ones((k2, k2), np.uint8)).astype(bool)
    inner = boundary_prf(e_ren & interior, e_ref & interior, tol)

    metrics = {
        "f1": round(allm["f1"], 6),
        "precision": round(allm["precision"], 6),
        "recall": round(allm["recall"], 6),
        "chamfer": round(allm["chamfer_px"] / pair.size, 6),
        "interior_f1": round(inner["f1"], 6),
        "tolerance_px": round(tol, 3),
        "edge_pixels_reference": int(e_ref.sum()),
        "edge_pixels_render": int(e_ren.sum()),
    }
    res = EvalResult(NAME, float(allm["f1"]), metrics)
    if cfg.debug:
        vis = np.full(e_ref.shape + (3,), 0.1, np.float32)
        vis[e_ref] = (0.3, 0.9, 0.4)  # reference edges: green
        vis[e_ren] = (0.9, 0.35, 0.9)  # render edges: magenta
        vis[e_ref & e_ren] = (1.0, 1.0, 1.0)  # exact overlap: white
        res.artifacts["edges"] = vis
    return res
