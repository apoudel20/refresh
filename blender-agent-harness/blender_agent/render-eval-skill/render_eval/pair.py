"""Load a reference image and a render, find the object in each, and align them.

Every eval works on an :class:`ImagePair`: two ``size x size`` RGB images plus a
foreground mask for each. Preprocessing does three things:

1. **Foreground mask.** Taken from the alpha channel when the image has real
   transparency (Blender renders with Film > Transparent), otherwise from a
   background-removal model (rembg), or from a mask file you pass in.
2. **Framing alignment** (``align="bbox"``). Each image is cropped to a square around
   its own foreground bounding box, so differences in framing, zoom and position
   do not count against the render. Aspect ratio is preserved.
3. **Background normalisation** (``background="neutral"``). Both foregrounds are
   composited onto the same flat grey, so the reference photo's backdrop does not
   count as a mismatch.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from render_eval.base import EvalConfig

ImageInput = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class LoadedImage:
    rgb: np.ndarray  # H x W x 3 float32 in [0, 1]
    alpha: np.ndarray  # H x W float32 soft matte in [0, 1]
    mask_source: str  # "alpha" | "segmentation" | "provided" | "none"
    name: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImagePair:
    """Aligned reference/render pair. All arrays are ``size x size``."""

    ref: np.ndarray  # float32 RGB [0, 1]
    ren: np.ndarray
    ref_alpha: np.ndarray  # float32 soft matte [0, 1]
    ren_alpha: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)  # evals share expensive intermediates here

    @property
    def ref_mask(self) -> np.ndarray:
        return self.ref_alpha >= 0.5

    @property
    def ren_mask(self) -> np.ndarray:
        return self.ren_alpha >= 0.5

    @property
    def union(self) -> np.ndarray:
        return self.ref_mask | self.ren_mask

    @property
    def intersection(self) -> np.ndarray:
        return self.ref_mask & self.ren_mask

    @property
    def size(self) -> int:
        return int(self.ref.shape[0])

    def pil(self, which: str) -> Image.Image:
        arr = self.ref if which == "ref" else self.ren
        return Image.fromarray(to_uint8(arr))


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Loading and masks
# --------------------------------------------------------------------------- #


def _open(src: ImageInput) -> tuple[Image.Image, str]:
    if isinstance(src, Image.Image):
        return src, "<PIL.Image>"
    if isinstance(src, np.ndarray):
        arr = src
        if arr.dtype != np.uint8:
            arr = to_uint8(arr.astype(np.float32))
        return Image.fromarray(arr), "<ndarray>"
    path = Path(src).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {src}")
    img = Image.open(path)
    img.load()
    return ImageOps.exif_transpose(img), str(path)


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def _clean_mask(alpha: np.ndarray, min_rel_area: float = 0.02) -> np.ndarray:
    """Drop specks: keep connected components at least ``min_rel_area`` of the largest."""
    binary = (alpha >= 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_ids = 1 + np.flatnonzero(areas >= areas.max() * min_rel_area)
    keep = np.isin(labels, keep_ids)
    # Keep the soft edge around kept components, zero everything near dropped ones.
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return np.where(keep, alpha, 0.0).astype(np.float32)


_MATTE_CACHE: dict[tuple[str, tuple[int, ...], str], np.ndarray] = {}


def segment_foreground(rgb_u8: np.ndarray, model_name: str) -> np.ndarray:
    """Soft foreground matte in [0, 1] from rembg.

    Cached by image content, so a reference compared against many candidates is segmented once.
    The default model is BiRefNet-lite: isnet-general-use lost a white dog's body on a light grey
    backdrop, and BiRefNet keeps fur detail that u2net smooths away.
    """
    key = (hashlib.sha1(rgb_u8.tobytes()).hexdigest(), rgb_u8.shape, model_name)
    if key not in _MATTE_CACHE:
        from rembg import remove

        from render_eval.models import rembg_session

        matte = remove(Image.fromarray(rgb_u8), session=rembg_session(model_name), only_mask=True)
        if len(_MATTE_CACHE) >= 32:
            _MATTE_CACHE.pop(next(iter(_MATTE_CACHE)))
        _MATTE_CACHE[key] = np.asarray(matte.convert("L"), dtype=np.float32) / 255.0
    return _MATTE_CACHE[key].copy()


def load_image(src: ImageInput, *, mask: ImageInput | None = None, mask_model: str = "birefnet-general-lite") -> LoadedImage:
    img, name = _open(src)
    warnings: list[str] = []
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb, alpha = arr[..., :3], arr[..., 3]
    h, w = alpha.shape

    if mask is not None:
        m_img, _ = _open(mask)
        m = m_img.convert("L")
        if m.size != (w, h):
            m = m.resize((w, h), Image.BILINEAR)
        alpha = np.asarray(m, dtype=np.float32) / 255.0
        source = "provided"
    elif _has_alpha(img) and 0.001 < float((alpha < 0.5).mean()) < 0.999:
        source = "alpha"
    else:
        alpha = segment_foreground((rgb * 255 + 0.5).astype(np.uint8), mask_model)
        source = "segmentation"

    alpha = _clean_mask(alpha)
    if float((alpha >= 0.5).mean()) < 1e-4:
        warnings.append(f"{name}: no foreground found ({source}); using the whole image as foreground")
        alpha = np.ones((h, w), np.float32)
        source = "none"
    return LoadedImage(rgb=rgb, alpha=alpha.astype(np.float32), mask_source=source, name=name, warnings=warnings)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


def _crop_square(img: LoadedImage, cfg: EvalConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h, w = img.alpha.shape
    bg = np.asarray(cfg.bg_color, np.float32)
    a = img.alpha[..., None]
    rgb = img.rgb * a + bg * (1.0 - a) if cfg.background == "neutral" else img.rgb

    ys, xs = np.nonzero(img.alpha >= 0.5)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    info = {
        "original_size": [w, h],
        "mask_source": img.mask_source,
        "bbox": [x0, y0, x1, y1],
        "bbox_aspect": round((x1 - x0) / max(1, y1 - y0), 4),
        "foreground_fraction": round(float((img.alpha >= 0.5).mean()), 4),
    }

    if cfg.align == "bbox":
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        side = int(math.ceil(max(x1 - x0, y1 - y0) * (1.0 + 2.0 * cfg.margin)))
    elif cfg.align == "none":
        cx, cy = w / 2.0, h / 2.0
        side = max(w, h)
    else:
        raise ValueError(f"unknown align mode {cfg.align!r} (use 'bbox' or 'none')")

    left, top = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, left + side - w), max(0, top + side - h)
    rgb_p = np.stack(
        [np.pad(rgb[..., c], ((pad_t, pad_b), (pad_l, pad_r)), constant_values=bg[c]) for c in range(3)], axis=-1
    )
    alpha_p = np.pad(img.alpha, ((pad_t, pad_b), (pad_l, pad_r)), constant_values=0.0)
    ox, oy = left + pad_l, top + pad_t
    rgb_c = rgb_p[oy : oy + side, ox : ox + side]
    alpha_c = alpha_p[oy : oy + side, ox : ox + side]

    s = cfg.size
    interp = cv2.INTER_AREA if side > s else cv2.INTER_CUBIC
    rgb_s = np.clip(cv2.resize(rgb_c, (s, s), interpolation=interp), 0.0, 1.0).astype(np.float32)
    alpha_s = np.clip(cv2.resize(alpha_c, (s, s), interpolation=cv2.INTER_LINEAR), 0.0, 1.0).astype(np.float32)
    info["crop_box"] = [left, top, left + side, top + side]
    info["scale"] = round(s / side, 5)
    return rgb_s, alpha_s, info


def make_pair(
    reference: ImageInput,
    render: ImageInput,
    cfg: EvalConfig | None = None,
    *,
    reference_mask: ImageInput | None = None,
    render_mask: ImageInput | None = None,
) -> ImagePair:
    cfg = cfg or EvalConfig()
    ref_img = load_image(reference, mask=reference_mask, mask_model=cfg.mask_model)
    ren_img = load_image(render, mask=render_mask, mask_model=cfg.mask_model)
    ref, ref_a, ref_info = _crop_square(ref_img, cfg)
    ren, ren_a, ren_info = _crop_square(ren_img, cfg)
    meta = {
        "reference": {"source": ref_img.name, **ref_info},
        "render": {"source": ren_img.name, **ren_info},
        "size": cfg.size,
        "align": cfg.align,
        "background": cfg.background,
        "warnings": ref_img.warnings + ren_img.warnings,
    }
    return ImagePair(ref=ref, ren=ren, ref_alpha=ref_a, ren_alpha=ren_a, meta=meta)
