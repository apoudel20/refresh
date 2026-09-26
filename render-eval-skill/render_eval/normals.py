"""Eval 3: surface-normal matching.

Estimates a per-pixel surface normal map for both images and measures the angle
between them where both objects are present. This is the most direct image-space
test of local shape: bumps, creases and curvature show up here even when the
silhouette already matches.

Backends:

* ``marigold`` (default): Marigold normals v1.1, a diffusion-based normal
  estimator. Accurate; about 7 s per pair on an M3 Max, 2 GB of weights.
* ``depth``: normals from the gradients of the Depth Anything depth maps (shared
  with the depth eval, so nearly free). Assumes the object's depth range is about
  half its image width, so absolute angles are approximate, but the same
  assumption applies to both images.

Metrics follow the normal-estimation literature: mean and median angular error in
degrees, and the fraction of pixels within 11.25, 22.5 and 30 degrees.

score = fraction of pixels within 22.5 degrees. (1 - mean_angle / 90 compresses badly:
pure noise still scores about 0.6 because most estimated normals face the camera.)
"""

from __future__ import annotations

import cv2
import numpy as np

from render_eval._geometry import colorize, hstack
from render_eval.base import EvalConfig, EvalResult, clip01
from render_eval.pair import ImagePair

NAME = "normals"
DESCRIPTION = "Surface-normal maps (Marigold or depth-derived) compared by angular error"


def _unit(n: np.ndarray) -> np.ndarray:
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)


def normals_from_depth(d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normals from relative inverse depth (larger = closer), as unit vectors facing the camera."""
    region = mask if mask.any() else np.ones_like(mask)
    lo, hi = np.percentile(d[region], [2, 98])
    size = d.shape[0]
    height = (d - lo) / max(hi - lo, 1e-8) * 0.5 * size  # height toward camera, in pixels
    height = cv2.GaussianBlur(height.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return _unit(np.stack([-gx, -gy, np.ones_like(gx)], axis=-1))


def predict_normals(pair: ImagePair, cfg: EvalConfig) -> tuple[np.ndarray, np.ndarray]:
    key = ("normals", cfg.normals_backend, cfg.normals_model)
    if key in pair.cache:
        return pair.cache[key]

    if cfg.normals_backend == "marigold":
        import torch

        from render_eval.models import marigold_normals, pick_device

        pipe = marigold_normals(cfg.normals_model, pick_device(cfg.device))
        # Same seed for each image, so both start from identical diffusion noise: identical
        # inputs give identical normals and the noise does not show up as a shape difference.
        pred = np.concatenate(
            [
                np.asarray(pipe(pair.pil(w), generator=torch.Generator("cpu").manual_seed(0)).prediction, np.float32)
                for w in ("ref", "ren")
            ]
        )  # (2, H, W, 3) in [-1, 1]
        if pred.shape[1:3] != (pair.size, pair.size):
            pred = np.stack([cv2.resize(p, (pair.size, pair.size), interpolation=cv2.INTER_LINEAR) for p in pred])
        ref_n, ren_n = _unit(pred[0]), _unit(pred[1])
    elif cfg.normals_backend == "depth":
        from render_eval.depth import predict_depth

        ref_d, ren_d = predict_depth(pair, cfg)
        ref_n, ren_n = normals_from_depth(ref_d, pair.ref_mask), normals_from_depth(ren_d, pair.ren_mask)
    else:
        raise ValueError(f"unknown normals backend {cfg.normals_backend!r} (use 'marigold' or 'depth')")

    pair.cache[key] = (ref_n, ren_n)
    return pair.cache[key]


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=-1), -1.0, 1.0)))


def evaluate(pair: ImagePair, cfg: EvalConfig) -> EvalResult:
    ref_n, ren_n = predict_normals(pair, cfg)
    # Skip a thin band at the outlines: normals there are unreliable and belong to the silhouette eval.
    region = cv2.erode(pair.intersection.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if region.sum() < 0.005 * pair.size**2:
        region = pair.union
    err = angular_error_deg(ref_n, ren_n)
    pair.cache["normals_error"] = (err, region)  # reused by render_eval.vectorize
    e = err[region]
    metrics = {
        "mean_angle_deg": round(float(e.mean()), 4),
        "median_angle_deg": round(float(np.median(e)), 4),
        "within_11_25": round(float((e < 11.25).mean()), 6),
        "within_22_5": round(float((e < 22.5).mean()), 6),
        "within_30": round(float((e < 30.0).mean()), 6),
        "pixels": int(region.sum()),
        "backend": cfg.normals_backend,
    }
    res = EvalResult(NAME, clip01(float((e < 22.5).mean())), metrics)
    if cfg.debug:
        def vis(n: np.ndarray, m: np.ndarray) -> np.ndarray:
            return np.where(m[..., None], (n + 1.0) / 2.0, 0.15).astype(np.float32)

        res.artifacts["normals"] = hstack(vis(ref_n, pair.ref_mask), vis(ren_n, pair.ren_mask), colorize(err, region))
    return res
