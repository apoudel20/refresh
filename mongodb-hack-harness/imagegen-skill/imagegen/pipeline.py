"""Model-driven image operations: generate, edit (whole image or a region), upscale, make seamless.

Every function takes a ``backend`` (see :mod:`imagegen.backends`) and PIL images,
and returns PIL images, so they chain freely. Region edits never trust the model
with pixels outside the region: the model sees a context crop, and its output is
composited back through a feathered mask.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from PIL import Image

from imagegen import ops
from imagegen.backends import Backend, GenerationResult
from imagegen.ops import Box, ImageLike

Log = Callable[[str], None]


def _nolog(_: str) -> None:
    pass


@dataclass
class EditResult:
    image: Image.Image
    region: Box | None = None
    context_box: Box | None = None
    generations: list[GenerationResult] = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        return {
            "size": list(self.image.size),
            "region": list(self.region) if self.region else None,
            "context_box": list(self.context_box) if self.context_box else None,
            "calls": [{"backend": g.backend, "model": g.model, **g.meta, "usage": g.usage} for g in self.generations],
        }


# --------------------------------------------------------------------------- #
# Generate
# --------------------------------------------------------------------------- #


def generate(
    prompt: str,
    *,
    backend: Backend,
    references: Sequence[ImageLike] = (),
    size: tuple[int, int] | None = None,
    aspect_ratio: str | None = None,
    n: int = 1,
    fit: str = "cover",
    resample: str = "lanczos",
    **options: Any,
) -> tuple[list[Image.Image], GenerationResult]:
    """Text (+ optional reference images) -> image(s). ``size`` forces exact output pixels."""
    if size and not aspect_ratio:
        aspect_ratio = ops.nearest_aspect(size[0], size[1], backend.aspect_ratios)
    res = backend.generate(prompt, references=[ops.load_image(r) for r in references], aspect_ratio=aspect_ratio, n=n, **options)
    images = [ops.fit(im, size, fit, resample) if size else im for im in res.images]
    return images, res


# --------------------------------------------------------------------------- #
# Edit
# --------------------------------------------------------------------------- #

_EDIT_FULL = """\
Edit image 1. Instruction: {instruction}
Keep everything the instruction does not mention exactly as it is: same framing, scale,
composition, colours, lighting and style. Output the full edited image at the same aspect ratio.{extra}"""

_EDIT_REGION = """\
Image 1 is the image to edit. Image 2 is the same image with the area you may change tinted
red and outlined.
Change ONLY the marked area. Instruction for that area: {instruction}
Outside the marked area, reproduce image 1 exactly (same framing, scale, colours, lighting).
The new content must blend seamlessly into its surroundings: continue textures, edges, lighting
and perspective across the boundary. Output image 1 with the edit applied — no red tint, no outline.{extra}"""


def edit(
    image: ImageLike,
    instruction: str,
    *,
    backend: Backend,
    region: Box | None = None,
    mask: ImageLike | None = None,
    context: float = 0.5,
    feather: float | None = None,
    references: Sequence[ImageLike] = (),
    color_match: bool = True,
    log: Log = _nolog,
) -> EditResult:
    """Edit a whole image, or only ``region`` (x0,y0,x1,y1) / ``mask`` (white = editable).

    ``context`` is how much surrounding image (as a fraction of the region's larger side)
    the model gets to see around a region, so its edit matches the neighbourhood.
    """
    base = ops.load_image(image)
    extra = _refs_note(references, first_index=2 if region is None and mask is None else 3)

    if region is None and mask is None:
        res = backend.generate(
            _EDIT_FULL.format(instruction=instruction, extra=extra),
            references=[base, *map(ops.load_image, references)],
            aspect_ratio=ops.nearest_aspect(*base.size, backend.aspect_ratios),
        )
        out = ops.fit(res.image, base.size, "stretch").convert(base.mode)
        return EditResult(image=out, generations=[res])

    m = ops.load_mask(mask, base.size) if mask is not None else ops.box_mask(base.size, region)
    bbox = ops.mask_bbox(m)
    if bbox is None:
        raise ValueError("Edit mask is empty")
    rw, rh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ctx = ops.expand_box(bbox, int(round(max(rw, rh) * context)), base.size)
    aspect = ops.nearest_aspect(ctx[2] - ctx[0], ctx[3] - ctx[1], backend.aspect_ratios)
    ctx = ops.fit_box_to_aspect(ctx, ops.ratio_value(aspect), base.size)
    log(f"edit region {bbox} with context {ctx} ({aspect})")

    crop = base.crop(ctx).convert("RGB")
    crop_mask = m.crop(ctx)
    res = backend.generate(
        _EDIT_REGION.format(instruction=instruction, extra=extra),
        references=[crop, ops.highlight(crop, crop_mask), *map(ops.load_image, references)],
        aspect_ratio=aspect,
    )
    patch = ops.fit(res.image.convert("RGB"), crop.size, "stretch")
    if color_match:
        # Calibrate on pixels the model was told to keep: the ring just outside the mask.
        band = ops.ring(crop_mask, max(4, min(crop.size) // 12))
        patch = ops.match_colors(patch, crop, band)
    radius = feather if feather is not None else max(1.0, min(rw, rh) * 0.04)
    soft = ops.feather(m, radius)
    full_patch = base.convert("RGB").copy()
    full_patch.paste(patch, ctx[:2])
    out = ops.composite(base, full_patch.crop(ctx), soft, ctx)
    return EditResult(image=out, region=bbox, context_box=ctx, generations=[res])


def _refs_note(references: Sequence[ImageLike], first_index: int) -> str:
    if not references:
        return ""
    last = first_index + len(references) - 1
    span = f"image {first_index}" if last == first_index else f"images {first_index}-{last}"
    return f"\n{span.capitalize()} are extra references for style and content only; do not copy their framing."


# --------------------------------------------------------------------------- #
# Upscale
# --------------------------------------------------------------------------- #

_UPSCALE = """\
Image 1 is a crop of a larger image that was enlarged with a basic resize, so it looks soft.
Re-render image 1 at high fidelity: crisp edges and fine, natural detail that fits the material
and art style. Do NOT change the composition, positions, shapes, colours or lighting; do not add
or remove anything; keep exactly the same framing and aspect ratio.{hint}"""


def upscale(
    image: ImageLike,
    scale: float = 2.0,
    *,
    method: str = "lanczos",
    backend: Backend | None = None,
    tile: int = 1024,
    overlap: int = 128,
    hint: str = "",
    concurrency: int = 4,
    log: Log = _nolog,
) -> Image.Image:
    """Enlarge by ``scale``. ``lanczos``/``bicubic``/``nearest`` are local and free;
    ``ai`` resizes with Lanczos first, then has the model re-detail overlapping tiles
    (colour-matched and blended), so any output size works and layout can't drift.
    """
    base = ops.load_image(image)
    target = (max(1, round(base.width * scale)), max(1, round(base.height * scale)))
    if method != "ai":
        return ops.resize(base, target, method)
    if backend is None:
        raise ValueError("method='ai' needs a backend")

    alpha = base.getchannel("A").resize(target, Image.Resampling.LANCZOS) if base.mode == "RGBA" else None
    soft = ops.resize(base.convert("RGB"), target, "lanczos")
    boxes = ops.tile_boxes(target, tile, overlap)
    log(f"ai upscale {base.size} -> {target} in {len(boxes)} tile(s)")
    prompt = _UPSCALE.format(hint=f"\nContext: {hint}" if hint else "")

    def work(box: Box) -> tuple[Box, Image.Image]:
        src = soft.crop(box)
        aspect = ops.nearest_aspect(*src.size, backend.aspect_ratios)
        res = backend.generate(prompt, references=[src], aspect_ratio=aspect)
        out = ops.fit(res.image.convert("RGB"), src.size, "stretch")
        log(f"  tile {box} done")
        return box, ops.match_colors(out, src)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        tiles = list(pool.map(work, boxes))
    out = ops.blend_tiles(target, tiles, "RGB")
    if alpha is not None:
        out.putalpha(alpha)
    return out


# --------------------------------------------------------------------------- #
# Seamless / tileable
# --------------------------------------------------------------------------- #

_SEAM = """\
This texture was offset so its wrap-around seams now run through the middle as a cross.
Repaint the marked cross so the texture continues naturally across it with no visible seam,
line, or change in scale, colour or lighting. {hint}"""


def seamless(
    image: ImageLike,
    *,
    method: str = "blend",
    backend: Backend | None = None,
    band: float = 0.2,
    hint: str = "",
    log: Log = _nolog,
) -> Image.Image:
    """Make a texture tile seamlessly.

    ``blend``: local cross-fade with a half-offset copy (free, fine for organic textures,
    can ghost on strongly structured ones). ``ai``: offset by half so the seams form a
    cross in the centre, have the model repaint only that cross, then offset back —
    the borders are original interior pixels, so the result wraps exactly.
    """
    img = ops.load_image(image)
    if method == "blend":
        return ops.seamless_blend(img, band=band)
    if backend is None:
        raise ValueError("method='ai' needs a backend")
    W, H = img.size
    rolled = ops.roll(img, W // 2, H // 2)
    res = edit(
        rolled,
        _SEAM.format(hint=hint).strip(),
        backend=backend,
        mask=ops.cross_mask(img.size, band),
        context=0.0,
        feather=max(2.0, min(W, H) * band * 0.15),
        log=log,
    )
    return ops.roll(res.image, -(W // 2), -(H // 2))
