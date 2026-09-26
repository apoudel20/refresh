"""Local, deterministic pixel operations used by the generation pipeline.

Nothing in this module calls a model: regions and masks, compositing, resizing,
tiled blending, seamless-texture helpers, chroma keying, and atlas grid
packing/slicing. Boxes use PIL's ``(left, top, right, bottom)`` convention.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ImageLike = Union[str, Path, bytes, Image.Image]
Box = tuple[int, int, int, int]

RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
}


# --------------------------------------------------------------------------- #
# I/O and parsing
# --------------------------------------------------------------------------- #


def load_image(src: ImageLike) -> Image.Image:
    """Open a path / bytes / PIL image (fully loaded, so the file handle is released)."""
    if isinstance(src, Image.Image):
        return src
    if isinstance(src, (bytes, bytearray)):
        im = Image.open(io.BytesIO(bytes(src)))
    else:
        p = Path(src).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {src}")
        im = Image.open(p)
    im.load()
    return im


def save_image(img: Image.Image, path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".jpg", ".jpeg") and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(p)
    return p


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_to_url(src: ImageLike) -> str:
    """A ``data:`` URL (or pass-through http(s)/data URL) for sending an image to an API."""
    import base64

    if isinstance(src, str) and src.startswith(("http://", "https://", "data:")):
        return src
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(load_image(src))).decode("ascii")


def parse_size(text: str | int | Sequence[int]) -> tuple[int, int]:
    """``"512"`` -> (512, 512); ``"512x256"`` -> (512, 256); ints and pairs pass through."""
    if isinstance(text, int):
        return text, text
    if not isinstance(text, str):
        w, h = text
        return int(w), int(h)
    parts = text.lower().replace("*", "x").split("x")
    if len(parts) == 1:
        return int(parts[0]), int(parts[0])
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError(f"Bad size {text!r}; expected N or WxH")


def parse_grid(text: str | Sequence[int]) -> tuple[int, int]:
    """``"4x2"`` -> (columns=4, rows=2)."""
    return parse_size(text)


def parse_region(text: str, size: tuple[int, int]) -> Box:
    """``"x,y,w,h"`` in pixels, or all-fractional (``"0.25,0.25,0.5,0.5"``) relative to ``size``."""
    vals = [float(v) for v in text.replace(" ", "").split(",")]
    if len(vals) != 4:
        raise ValueError(f"Bad region {text!r}; expected x,y,w,h")
    if all(0 <= v <= 1 for v in vals) and any("." in s for s in text.split(",")):
        W, H = size
        vals = [vals[0] * W, vals[1] * H, vals[2] * W, vals[3] * H]
    x, y, w, h = (int(round(v)) for v in vals)
    if w <= 0 or h <= 0:
        raise ValueError(f"Region {text!r} has no area")
    return clamp_box((x, y, x + w, y + h), size)


def parse_color(text: str | Sequence[int]) -> tuple[int, int, int]:
    if not isinstance(text, str):
        r, g, b = text[:3]
        return int(r), int(g), int(b)
    t = text.strip().lstrip("#")
    if "," in t:
        r, g, b = (int(v) for v in t.split(","))
        return r, g, b
    if len(t) == 3:
        t = "".join(c * 2 for c in t)
    if len(t) != 6:
        raise ValueError(f"Bad color {text!r}; expected #rrggbb or r,g,b")
    return int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)


# --------------------------------------------------------------------------- #
# Boxes, masks, compositing
# --------------------------------------------------------------------------- #


def clamp_box(box: Box, size: tuple[int, int]) -> Box:
    W, H = size
    l, t, r, b = box
    l, r = max(0, min(l, W)), max(0, min(r, W))
    t, b = max(0, min(t, H)), max(0, min(b, H))
    if r <= l or b <= t:
        raise ValueError(f"Box {box} does not overlap image of size {size}")
    return l, t, r, b


def box_mask(size: tuple[int, int], box: Box) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=255)
    return m


def load_mask(src: ImageLike, size: tuple[int, int]) -> Image.Image:
    """White (or opaque, for RGBA masks with real transparency) = editable."""
    m = load_image(src)
    if m.mode in ("RGBA", "LA") and m.getchannel("A").getextrema()[0] < 255:
        m = m.getchannel("A")
    else:
        m = m.convert("L")
    if m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    return m


def mask_bbox(mask: Image.Image) -> Box | None:
    return mask.point(lambda v: 255 if v >= 128 else 0).getbbox()


def expand_box(box: Box, pad: int, size: tuple[int, int]) -> Box:
    l, t, r, b = box
    return clamp_box((l - pad, t - pad, r + pad, b + pad), size)


def ratio_value(aspect: str) -> float:
    w, h = aspect.split(":")
    return float(w) / float(h)


def nearest_aspect(w: int, h: int, options: Sequence[str]) -> str:
    """Pick the aspect-ratio string closest (in log space) to ``w:h``."""
    target = math.log(w / h)
    opts = [o for o in options if o != "auto"]
    return min(opts, key=lambda o: abs(math.log(ratio_value(o)) - target))


def fit_box_to_aspect(box: Box, ratio: float, size: tuple[int, int]) -> Box:
    """Grow ``box`` (never shrink) toward ``ratio`` = w/h, keeping it centred and inside the image."""
    W, H = size
    l, t, r, b = box
    w, h = r - l, b - t
    if w / h < ratio:
        w = min(W, max(w, round(h * ratio)))
    else:
        h = min(H, max(h, round(w / ratio)))
    cx, cy = (l + r) / 2, (t + b) / 2
    l = int(round(min(max(cx - w / 2, 0), W - w)))
    t = int(round(min(max(cy - h / 2, 0), H - h)))
    return l, t, l + w, t + h


def feather(mask: Image.Image, radius: float) -> Image.Image:
    """Soften the mask edge *inward* only: the ramp lives inside the mask, so pixels outside stay exactly 0."""
    if radius <= 0:
        return mask
    grow = int(math.ceil(radius * 2))
    eroded = mask.filter(ImageFilter.MinFilter(2 * grow + 1))
    soft = eroded.filter(ImageFilter.GaussianBlur(radius))
    return Image.fromarray(np.minimum(np.asarray(soft), np.asarray(mask)), "L")


def composite(base: Image.Image, patch: Image.Image, mask: Image.Image, box: Box) -> Image.Image:
    """Blend ``patch`` (resized to ``box``) over ``base`` wherever ``mask`` (full-size, L) is set."""
    l, t, r, b = box
    patch = patch.convert(base.mode).resize((r - l, b - t), Image.Resampling.LANCZOS)
    region = base.crop(box)
    blended = Image.composite(patch, region, mask.crop(box))
    out = base.copy()
    out.paste(blended, (l, t))
    return out


def match_colors(src: Image.Image, ref: Image.Image, where: Image.Image | None = None) -> Image.Image:
    """Shift ``src``'s per-channel mean/std to ``ref``'s, measured over ``where`` (L mask) if given.

    Used to cancel the small global colour drift image models introduce, so a
    pasted-back patch or tile doesn't show a visible boundary.
    """
    mode = src.mode
    s = np.asarray(src.convert("RGB"), dtype=np.float64)
    r = np.asarray(ref.convert("RGB").resize(src.size), dtype=np.float64)
    sel = np.ones(s.shape[:2], bool) if where is None else np.asarray(where.resize(src.size)) >= 128
    if sel.sum() < 16:
        return src
    out = s.copy()
    for c in range(3):
        sm, ss = s[..., c][sel].mean(), s[..., c][sel].std() + 1e-6
        rm, rs = r[..., c][sel].mean(), r[..., c][sel].std() + 1e-6
        out[..., c] = (s[..., c] - sm) * min(rs / ss, 2.0) + rm
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    if mode == "RGBA":
        img.putalpha(src.getchannel("A"))
    return img


def ring(mask: Image.Image, width: int) -> Image.Image:
    """The band of pixels just outside ``mask`` (dilated minus original)."""
    size = max(3, width * 2 + 1)
    dil = mask.filter(ImageFilter.MaxFilter(size | 1))
    return Image.fromarray(((np.asarray(dil) >= 128) & (np.asarray(mask) < 128)).astype(np.uint8) * 255, "L")


def highlight(img: Image.Image, mask: Image.Image, color=(255, 0, 64)) -> Image.Image:
    """Tint and outline the masked area, as a visual pointer for the image model."""
    base = img.convert("RGB")
    tint = Image.new("RGB", base.size, color)
    hard = mask.point(lambda v: 255 if v >= 128 else 0)
    out = Image.composite(Image.blend(base, tint, 0.45), base, hard)
    edge = hard.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(max(3, (min(base.size) // 150) | 1)))
    out.paste(tint, (0, 0), edge)
    return out


# --------------------------------------------------------------------------- #
# Resizing and tiling
# --------------------------------------------------------------------------- #


def resize(img: Image.Image, size: tuple[int, int], method: str = "lanczos") -> Image.Image:
    return img if img.size == tuple(size) else img.resize(tuple(size), RESAMPLE[method])


def fit(img: Image.Image, size: tuple[int, int], mode: str = "cover", method: str = "lanczos") -> Image.Image:
    """Resize to exactly ``size``: ``cover`` (scale + centre crop), ``contain`` (letterbox), or ``stretch``."""
    W, H = size
    if img.size == (W, H) or mode == "stretch":
        return resize(img, (W, H), method)
    w, h = img.size
    scale = max(W / w, H / h) if mode == "cover" else min(W / w, H / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    scaled = img.resize((nw, nh), RESAMPLE[method])
    if mode == "cover":
        l, t = (nw - W) // 2, (nh - H) // 2
        return scaled.crop((l, t, l + W, t + H))
    canvas = Image.new(img.mode, (W, H))
    canvas.paste(scaled, ((W - nw) // 2, (H - nh) // 2))
    return canvas


def tile_boxes(size: tuple[int, int], tile: int, overlap: int) -> list[Box]:
    """Overlapping tiles covering ``size``; edge tiles are shifted inward rather than shrunk."""
    W, H = size

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        step = tile - overlap
        count = math.ceil((n - overlap) / step)
        return sorted({min(i * step, n - tile) for i in range(count)})

    return [(x, y, min(x + tile, W), min(y + tile, H)) for y in starts(H) for x in starts(W)]


def blend_tiles(size: tuple[int, int], tiles: Sequence[tuple[Box, Image.Image]], mode: str = "RGB") -> Image.Image:
    """Average overlapping tiles with linear ramps so no hard tile edges survive."""
    W, H = size
    chans = len(mode)
    acc = np.zeros((H, W, chans), np.float64)
    wsum = np.zeros((H, W, 1), np.float64)
    for (l, t, r, b), im in tiles:
        arr = np.asarray(im.convert(mode).resize((r - l, b - t)), np.float64).reshape(b - t, r - l, chans)
        wx = _ramp(r - l, l > 0, r < W)
        wy = _ramp(b - t, t > 0, b < H)
        w = (wy[:, None] * wx[None, :])[..., None]
        acc[t:b, l:r] += arr * w
        wsum[t:b, l:r] += w
    out = np.clip(acc / np.maximum(wsum, 1e-9), 0, 255).astype(np.uint8)
    return Image.fromarray(out.squeeze(-1) if chans == 1 else out, mode)


def _ramp(n: int, fade_start: bool, fade_end: bool) -> np.ndarray:
    w = np.ones(n)
    k = max(1, n // 4)
    ramp = (np.arange(k) + 1) / (k + 1)
    if fade_start:
        w[:k] = np.minimum(w[:k], ramp)
    if fade_end:
        w[-k:] = np.minimum(w[-k:], ramp[::-1])
    return w


# --------------------------------------------------------------------------- #
# Seamless textures
# --------------------------------------------------------------------------- #


def roll(img: Image.Image, dx: int, dy: int) -> Image.Image:
    arr = np.roll(np.asarray(img), (dy, dx), axis=(0, 1))
    return Image.fromarray(arr, img.mode)


def seam_score(img: Image.Image) -> float:
    """How visible the wrap-around seams are, relative to ordinary neighbouring-pixel variation.

    ~1.0 means the seams look like any other pixel boundary (tiles cleanly);
    values well above ~1.5 mean a visible seam when the texture repeats.
    """
    a = np.asarray(img.convert("RGB"), np.float64)
    interior = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2
    seam = (np.abs(a[:, 0] - a[:, -1]).mean() + np.abs(a[0, :] - a[-1, :]).mean()) / 2
    return float(seam / max(interior, 1e-6))


def seamless_blend(img: Image.Image, band: float = 0.25) -> Image.Image:
    """Make a texture tile by cross-fading each border band with its half-offset copy.

    Done one axis at a time: blending two horizontally-tileable images keeps the
    result horizontally tileable, so the vertical pass doesn't undo the first.
    The centre ``1 - 2*band`` of the image is untouched.
    """
    mode = img.mode
    a = np.asarray(img, np.float64)
    squeeze = a.ndim == 2
    if squeeze:
        a = a[..., None]
    H, W = a.shape[:2]
    for axis, n in ((1, W), (0, H)):
        d = np.minimum(np.arange(n) + 0.5, n - np.arange(n) - 0.5)  # distance to nearest edge
        w = np.clip(d / max(band * n, 1), 0, 1)
        w = w * w * (3 - 2 * w)  # smoothstep
        rolled = np.roll(a, n // 2, axis=axis)
        shape = (1, n, 1) if axis == 1 else (n, 1, 1)
        w = w.reshape(shape)
        a = w * a + (1 - w) * rolled
    out = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(out[..., 0] if squeeze else out, mode)


def cross_mask(size: tuple[int, int], band: float) -> Image.Image:
    """A centred '+' of relative width ``band`` — where the seams land after ``roll`` by half."""
    W, H = size
    bw, bh = max(2, round(W * band / 2)), max(2, round(H * band / 2))
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.rectangle((W // 2 - bw, 0, W // 2 + bw, H), fill=255)
    d.rectangle((0, H // 2 - bh, W, H // 2 + bh), fill=255)
    return m


def tile_preview(img: Image.Image, n: int = 2) -> Image.Image:
    w, h = img.size
    out = Image.new(img.mode, (w * n, h * n))
    for y in range(n):
        for x in range(n):
            out.paste(img, (x * w, y * h))
    return out


# --------------------------------------------------------------------------- #
# Transparency
# --------------------------------------------------------------------------- #


def chroma_key(img: Image.Image, color=(255, 0, 255), tolerance: float = 60, softness: float = 40) -> Image.Image:
    """Turn pixels near ``color`` transparent (soft edge over ``softness``) and de-spill the fringe."""
    rgb = np.asarray(img.convert("RGB"), np.float64)
    key = np.array(color, np.float64)
    dist = np.linalg.norm(rgb - key, axis=-1)
    alpha = np.clip((dist - tolerance) / max(softness, 1e-6), 0, 1)
    if img.mode == "RGBA":
        alpha = alpha * (np.asarray(img.getchannel("A"), np.float64) / 255)
    # Pull the key colour back out of semi-transparent fringe pixels.
    fringe = (alpha > 0) & (alpha < 1)
    a3 = alpha[..., None]
    unmixed = (rgb - (1 - a3) * key) / np.maximum(a3, 1e-3)
    rgb = np.where(fringe[..., None], np.clip(unmixed, 0, 255), rgb)
    out = np.dstack([rgb, alpha * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


# --------------------------------------------------------------------------- #
# Atlas grids
# --------------------------------------------------------------------------- #


def extrude(img: Image.Image, pad: int, wrap: bool = False) -> Image.Image:
    """Pad by repeating edge pixels (or wrapping, for tileable cells) to stop mip/filter bleeding."""
    if pad <= 0:
        return img
    a = np.asarray(img)
    widths = ((pad, pad), (pad, pad)) + (((0, 0),) if a.ndim == 3 else ())
    return Image.fromarray(np.pad(a, widths, mode="wrap" if wrap else "edge"), img.mode)


def next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def pack_grid(
    images: Sequence[Image.Image],
    columns: int,
    cell: tuple[int, int],
    padding: int = 0,
    wrap: Sequence[bool] | None = None,
    power_of_two: bool = False,
    mode: str = "RGBA",
) -> tuple[Image.Image, list[Box]]:
    """Lay cells out row-major on a grid. Returns the atlas and each cell's (x, y, w, h) frame."""
    cw, ch = cell
    rows = math.ceil(len(images) / columns)
    sw, sh = cw + 2 * padding, ch + 2 * padding
    W, H = columns * sw, rows * sh
    if power_of_two:
        W, H = next_pow2(W), next_pow2(H)
    atlas = Image.new(mode, (W, H), (0, 0, 0, 0) if mode == "RGBA" else 0)
    frames: list[Box] = []
    for i, im in enumerate(images):
        c, r = i % columns, i // columns
        im = fit(im.convert(mode), (cw, ch))
        slot = extrude(im, padding, wrap=bool(wrap and wrap[i]))
        atlas.paste(slot, (c * sw, r * sh))
        frames.append((c * sw + padding, r * sh + padding, cw, ch))
    return atlas, frames


def slice_grid(img: Image.Image, columns: int, rows: int, inset: int = 0) -> list[Image.Image]:
    """Cut an image into ``columns x rows`` equal cells (row-major), trimming ``inset`` px per side."""
    W, H = img.size
    out = []
    for r in range(rows):
        for c in range(columns):
            l, t = round(c * W / columns), round(r * H / rows)
            rr, bb = round((c + 1) * W / columns), round((r + 1) * H / rows)
            out.append(img.crop((l + inset, t + inset, rr - inset, bb - inset)))
    return out


def grid_guide(
    columns: int,
    rows: int,
    cell: tuple[int, int],
    labels: Sequence[str] | None = None,
    canvas: tuple[int, int] | None = None,
) -> tuple[Image.Image, Box]:
    """A layout reference: numbered grid cells, optionally centred on a larger neutral canvas.

    Returns the guide and the grid's box within it (to crop the generated sheet back out).
    """
    gw, gh = columns * cell[0], rows * cell[1]
    CW, CH = canvas or (gw, gh)
    img = Image.new("RGB", (CW, CH), (128, 128, 128))
    ox, oy = (CW - gw) // 2, (CH - gh) // 2
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(12, min(cell) // 6))
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    line = max(2, min(cell) // 64)
    for i in range(columns * rows):
        c, r = i % columns, i // columns
        x0, y0 = ox + c * cell[0], oy + r * cell[1]
        shade = 235 if (c + r) % 2 == 0 else 215
        d.rectangle((x0, y0, x0 + cell[0] - 1, y0 + cell[1] - 1), fill=(shade,) * 3, outline=(40, 40, 40), width=line)
        text = str(i + 1) if not labels or i >= len(labels) else f"{i + 1}: {labels[i]}"
        d.text((x0 + cell[0] // 2, y0 + cell[1] // 2), text, fill=(40, 40, 40), anchor="mm", font=font)
    return img, (ox, oy, ox + gw, oy + gh)
