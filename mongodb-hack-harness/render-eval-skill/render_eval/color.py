"""Eval 7: color and material palette.

Compares the colors of the two objects (foreground pixels only) in CIELAB, where
Euclidean distance approximates perceived difference (Delta E; about 2.3 is just
noticeable, 10+ is obvious).

* ``palette_emd``: Earth mover's distance between the two images' color
  distributions over one shared palette (k-means on both images' pixels, 8 colors).
  Each image is described by its area share of each palette color. Captures both
  which colors and how much of each. A shared palette matters: with separate
  per-image palettes, k-means puts its centers in different places and identical
  distributions come out about Delta E 6 apart.
* ``palette_emd_ab``: same, ignoring lightness, so it is less sensitive to
  lighting differences between the photo and the render.
* ``mean_delta_e2000``: CIEDE2000 difference between the average colors.
* ``wasserstein_L`` / ``_a`` / ``_b``: 1-D distribution distance per channel.

The palettes themselves (hex and area share) are in ``details``.

score = exp(-palette_emd / 20)   (Delta E 5 -> 0.78, 10 -> 0.61, 20 -> 0.37)
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

from render_eval.base import EvalConfig, EvalResult
from render_eval.pair import ImagePair

NAME = "color"
DESCRIPTION = "Foreground color palettes compared in CIELAB (EMD, Delta E)"

PALETTE_SIZE = 8
MAX_PIXELS = 20_000


def foreground_lab(img: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Lab values of foreground pixels, skipping the outline where colors blend with the background."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    sel = inner if inner.sum() >= 64 else mask
    px = img[sel].astype(np.float64)
    if len(px) > MAX_PIXELS:
        px = px[np.random.default_rng(seed).choice(len(px), MAX_PIXELS, replace=False)]
    return rgb2lab(px.reshape(-1, 1, 3)).reshape(-1, 3)


def shared_palette(lab_a: np.ndarray, lab_b: np.ndarray, k: int = PALETTE_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One k-means palette fitted to both pixel sets (equal samples from each).

    Returns (centers, weights_a, weights_b), sorted by combined share, where each
    weights vector is that image's area share of every palette color.
    """
    rng = np.random.default_rng(0)
    n = min(len(lab_a), len(lab_b))
    both = np.concatenate([lab_a[rng.choice(len(lab_a), n, replace=False)], lab_b[rng.choice(len(lab_b), n, replace=False)]])
    k = max(1, min(k, len(np.unique(np.round(both, 1), axis=0))))
    centers, labels = kmeans2(both, k, seed=0, minit="++")
    centers = centers[np.bincount(labels, minlength=k) > 0]

    def shares(lab: np.ndarray) -> np.ndarray:
        nearest = np.argmin(np.linalg.norm(lab[:, None, :] - centers[None, :, :], axis=-1), axis=1)
        return np.bincount(nearest, minlength=len(centers)).astype(np.float64) / len(lab)

    wa, wb = shares(lab_a), shares(lab_b)
    order = np.argsort(-(wa + wb))
    return centers[order], wa[order], wb[order]


def emd(c1: np.ndarray, w1: np.ndarray, c2: np.ndarray, w2: np.ndarray) -> float:
    """Exact earth mover's distance between two weighted point sets (weights sum to 1)."""
    n, m = len(c1), len(c2)
    cost = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=-1).ravel()
    a_eq = np.zeros((n + m, n * m))
    for i in range(n):
        a_eq[i, i * m : (i + 1) * m] = 1.0
    for j in range(m):
        a_eq[n + j, j::m] = 1.0
    res = linprog(cost, A_eq=a_eq, b_eq=np.concatenate([w1, w2]), bounds=(0, None), method="highs")
    return float(res.fun)


def _hex(lab_color: np.ndarray) -> str:
    rgb = np.clip(lab2rgb(lab_color.reshape(1, 1, 3)).reshape(3), 0, 1)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


def _swatches(centers: np.ndarray, weights: np.ndarray, width: int, height: int = 48) -> np.ndarray:
    bar = np.zeros((height, width, 3), np.float32)
    x = 0
    last = int(np.flatnonzero(weights > 0).max()) if (weights > 0).any() else -1
    for i, (c, w) in enumerate(zip(centers, weights)):
        x1 = width if i == last else x + int(round(w * width))
        bar[:, x:x1] = np.clip(lab2rgb(c.reshape(1, 1, 3)).reshape(3), 0, 1)
        x = x1
    return bar


def evaluate(pair: ImagePair, cfg: EvalConfig) -> EvalResult:
    lab_ref = foreground_lab(pair.ref, pair.ref_mask)
    lab_ren = foreground_lab(pair.ren, pair.ren_mask)
    centers, w_ref, w_ren = shared_palette(lab_ref, lab_ren)

    d_lab = emd(centers, w_ref, centers, w_ren)
    d_ab = emd(centers[:, 1:], w_ref, centers[:, 1:], w_ren)
    mean_de = float(deltaE_ciede2000(lab_ref.mean(0), lab_ren.mean(0)))
    metrics: dict[str, Any] = {
        "palette_emd": round(d_lab, 4),
        "palette_emd_ab": round(d_ab, 4),
        "mean_delta_e2000": round(mean_de, 4),
    }
    for i, ch in enumerate("Lab"):
        metrics[f"wasserstein_{ch}"] = round(float(wasserstein_distance(lab_ref[:, i], lab_ren[:, i])), 4)

    def listing(w: np.ndarray) -> list[dict[str, Any]]:
        order = np.argsort(-w)
        return [{"hex": _hex(centers[i]), "share": round(float(w[i]), 4)} for i in order if w[i] >= 0.005]

    details = {"reference_palette": listing(w_ref), "render_palette": listing(w_ren)}
    res = EvalResult(NAME, float(math.exp(-d_lab / 20.0)), metrics, details)
    if cfg.debug:
        gap = np.ones((6, pair.size, 3), np.float32)
        res.artifacts["color"] = np.concatenate(
            [_swatches(centers, w_ref, pair.size), gap, _swatches(centers, w_ren, pair.size)], axis=0
        )
    return res
