"""Eval 1: pixel similarity (PSNR, SSIM, LPIPS).

Compares the aligned, background-normalised images directly. Raw pixel error is
dominated by lighting and material differences, so LPIPS (a learned perceptual
distance) carries the most signal. SSIM and LPIPS are also reported over the
foreground only (union of both masks), because a large shared grey background
inflates whole-image similarity.

score = 0.5 * SSIM_fg + 0.5 * (1 - LPIPS_fg)
"""

from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity

from render_eval._geometry import colorize, hstack
from render_eval.base import EvalConfig, EvalResult, clip01
from render_eval.pair import ImagePair

NAME = "pixel"
DESCRIPTION = "PSNR, SSIM and LPIPS between the aligned images"


def _psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a - b) ** 2
    mse = float(diff[mask].mean()) if mask is not None and mask.any() else float(diff.mean())
    return 100.0 if mse <= 1e-10 else float(10.0 * math.log10(1.0 / mse))


def lpips_map(pair: ImagePair, cfg: EvalConfig) -> np.ndarray:
    import torch

    from render_eval.models import lpips_model, pick_device, to_tensor

    device = pick_device(cfg.device)
    model = lpips_model(cfg.lpips_net, device)
    with torch.no_grad():
        # LPIPS expects inputs in [-1, 1]; spatial=True returns a per-pixel map at input size.
        d = model(to_tensor(pair.ref, device) * 2 - 1, to_tensor(pair.ren, device) * 2 - 1)
    return d[0, 0].float().cpu().numpy()


def evaluate(pair: ImagePair, cfg: EvalConfig) -> EvalResult:
    fg = pair.union
    ssim_full, ssim_map = structural_similarity(pair.ref, pair.ren, channel_axis=2, data_range=1.0, full=True)
    ssim_map = ssim_map.mean(axis=2)
    ssim_fg = float(ssim_map[fg].mean()) if fg.any() else float(ssim_full)

    lp = lpips_map(pair, cfg)
    pair.cache["lpips_map"] = lp  # reused by render_eval.vectorize
    lpips_full = float(lp.mean())
    lpips_fg = float(lp[fg].mean()) if fg.any() else lpips_full

    metrics = {
        "psnr_db": round(_psnr(pair.ref, pair.ren), 4),
        "psnr_fg_db": round(_psnr(pair.ref, pair.ren, fg), 4),
        "ssim": round(float(ssim_full), 6),
        "ssim_fg": round(ssim_fg, 6),
        "lpips": round(lpips_full, 6),
        "lpips_fg": round(lpips_fg, 6),
        "lpips_net": cfg.lpips_net,
    }
    score = 0.5 * clip01(ssim_fg) + 0.5 * (1.0 - clip01(lpips_fg))
    res = EvalResult(NAME, clip01(score), metrics)
    if cfg.debug:
        err = np.abs(pair.ref - pair.ren).mean(axis=2)
        res.artifacts["pixel"] = hstack(pair.ref, pair.ren, colorize(err, fg), colorize(lp, fg))
    return res
