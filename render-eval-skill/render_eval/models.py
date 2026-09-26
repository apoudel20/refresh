"""Lazy, cached loaders for the local models the evals use.

Everything here downloads weights on first use (Hugging Face hub, rembg's model cache,
torchvision's cache) and then stays in memory for the rest of the process.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@functools.lru_cache(maxsize=4)
def rembg_session(model_name: str) -> Any:
    from rembg import new_session

    return new_session(model_name)


@functools.lru_cache(maxsize=2)
def lpips_model(net: str, device: str) -> Any:
    import lpips

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torchvision "pretrained" deprecation noise
        model = lpips.LPIPS(net=net, spatial=True, verbose=False)
    return model.to(device).eval()


@functools.lru_cache(maxsize=2)
def depth_model(model_id: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return processor, model


@functools.lru_cache(maxsize=1)
def marigold_normals(model_id: str, device: str) -> Any:
    import torch
    from diffusers import MarigoldNormalsPipeline

    dtype = torch.float32 if device == "cpu" else torch.float16
    kwargs: dict[str, Any] = {"dtype": dtype}
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"
    pipe = MarigoldNormalsPipeline.from_pretrained(model_id, **kwargs).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def to_tensor(img: np.ndarray, device: str) -> Any:
    """HxWx3 float [0,1] -> 1x3xHxW float tensor on device."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float().to(device)
