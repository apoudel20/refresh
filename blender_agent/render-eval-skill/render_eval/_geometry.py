"""Small geometry/visualisation helpers shared by several evals."""

from __future__ import annotations

import math

import cv2
import numpy as np


def bsds_tolerance(size: int) -> float:
    """Match tolerance used by the BSDS boundary benchmark: 0.75% of the image diagonal."""
    return max(1.5, 0.0075 * math.hypot(size, size))


def _dist_to(edges: np.ndarray) -> np.ndarray:
    """Distance (px) from every pixel to the nearest True pixel of ``edges``."""
    if not edges.any():
        return np.full(edges.shape, np.inf, np.float32)
    return cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 5)


def boundary_prf(pred: np.ndarray, gt: np.ndarray, tol: float) -> dict[str, float]:
    """Precision/recall/F1 of two boolean edge maps with a pixel tolerance, plus Chamfer distances.

    ``pred`` is the render's edge map and ``gt`` the reference's. Chamfer values are
    in pixels; callers normalise them by image size.
    """
    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    if n_pred == 0 and n_gt == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "chamfer_px": 0.0, "hd95_px": 0.0}
    if n_pred == 0 or n_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "chamfer_px": float("nan"), "hd95_px": float("nan")}
    d_to_gt = _dist_to(gt)[pred]
    d_to_pred = _dist_to(pred)[gt]
    precision = float((d_to_gt <= tol).mean())
    recall = float((d_to_pred <= tol).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    both = np.concatenate([d_to_gt, d_to_pred])
    return {
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
        "chamfer_px": float(0.5 * (d_to_gt.mean() + d_to_pred.mean())),
        "hd95_px": float(np.percentile(both, 95)),
    }


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide inner boundary of a boolean mask."""
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def colorize(values: np.ndarray, mask: np.ndarray | None = None, cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Float map -> RGB float [0,1] heatmap, normalised over ``mask`` (robust 2-98 percentile)."""
    v = values.astype(np.float32)
    region = v[mask] if mask is not None and mask.any() else v.ravel()
    lo, hi = np.percentile(region, [2, 98]) if region.size else (0.0, 1.0)
    norm = np.clip((v - lo) / max(hi - lo, 1e-8), 0, 1)
    rgb = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)[..., ::-1].astype(np.float32) / 255.0
    if mask is not None:
        rgb = np.where(mask[..., None], rgb, 0.15)
    return rgb


def hstack(*imgs: np.ndarray, gap: int = 4) -> np.ndarray:
    """Side-by-side panel of equally sized RGB float images with a white gap."""
    h = imgs[0].shape[0]
    sep = np.ones((h, gap, 3), np.float32)
    parts: list[np.ndarray] = []
    for i, im in enumerate(imgs):
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=-1)
        parts.append(im.astype(np.float32))
        if i < len(imgs) - 1:
            parts.append(sep)
    return np.concatenate(parts, axis=1)


def auto_canny(gray: np.ndarray, region: np.ndarray | None = None, high_pct: float = 90.0) -> np.ndarray:
    """Canny on a float image with thresholds from the gradient-magnitude distribution.

    The high threshold is the ``high_pct`` percentile of Sobel magnitude inside ``region``
    and the low threshold is 40% of it, so the result adapts to contrast and lighting.
    """
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, [0.5, 99.5])
    u8 = (np.clip((g - lo) / max(hi - lo, 1e-8), 0, 1) * 255).astype(np.uint8)
    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    sel = mag[region] if region is not None and region.any() else mag.ravel()
    sel = sel[sel > 0]
    high = float(np.percentile(sel, high_pct)) if sel.size else 100.0
    edges = cv2.Canny(u8, 0.4 * high, high, L2gradient=True).astype(bool)
    if region is not None:
        edges &= region
    return edges
