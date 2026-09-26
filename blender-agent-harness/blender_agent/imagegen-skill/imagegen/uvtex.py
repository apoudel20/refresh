"""Retexture a Blender object's UV texture atlas from a style reference, and check the result.

A UV atlas isn't a picture: each texel is glued to a spot on the mesh through the UV islands.
Image models don't know that. They shift, rescale or ignore the layout, leave holes, paint
lighting, and break colour continuity across UV seams. So this module:

1. **Retextures** by showing the model the current atlas, a copy with the islands outlined, and
   the style reference(s). Its output is then *clamped* to the islands: only texels inside an
   island are taken from the model, everything else keeps the original atlas. Island colours
   are bled outward into the padding margin, as Blender's bake margin does.
2. **Checks** the result against the atlas rules (``check``): same size and format, every island
   painted, padding bleed present, the model's content registered to the island boundaries (no
   drift), seam colour continuity, and UV-layout sanity from Blender.
3. **Repairs** failing islands one at a time with masked region edits (``repair``) and blends
   tone across broken seams (``fix_seams``).

UV data comes from ``imagegen/blender/uv_export.py`` (via ``imagegen uv export``): an island-id
map at texture resolution plus ``uv_info.json`` (islands, seams, textures, sanity counts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from imagegen import ops, pipeline
from imagegen.backends import Backend
from imagegen.ops import ImageLike

Log = Callable[[str], None]


def _nolog(_: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# UV layout
# --------------------------------------------------------------------------- #


@dataclass
class UVLayout:
    islands: np.ndarray  # int32 [H, W], 0 = no island
    overlap: np.ndarray  # uint8 [H, W]
    info: dict[str, Any]
    normals: np.ndarray | None = None  # float [H, W, 3] world-space surface normal per texel
    up_angle: np.ndarray | None = None  # float [H, W] where world-up points, degrees clockwise from image-up

    @classmethod
    def load(cls, path: str | Path) -> "UVLayout":
        root = Path(path)
        if root.is_file():
            root = root.parent
        data = np.load(root / "uv_data.npz")
        info = json.loads((root / "uv_info.json").read_text())
        normals = data["normals"].astype(np.float32) if "normals" in data.files else None
        up = data["up_angle"].astype(np.float32) if "up_angle" in data.files else None
        return cls(islands=data["islands"].astype(np.int32), overlap=data["overlap"], info=info, normals=normals, up_angle=up)

    @classmethod
    def from_islands(cls, islands: np.ndarray, seams: Sequence[dict] = (), **info: Any) -> "UVLayout":
        """Build a layout from an id map directly (tests, or masks made outside Blender)."""
        ids = sorted(int(k) for k in np.unique(islands) if k)
        isl = []
        for k in ids:
            ys, xs = np.nonzero(islands == k)
            isl.append({"id": k, "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                        "texels": int(len(xs)), "texel_density": None, "normal": None})  # fmt: skip
        H, W = islands.shape
        base = {"size": [W, H], "islands": isl, "seams": list(seams), "sanity": {"island_count": len(ids)}}
        return cls(islands=islands.astype(np.int32), overlap=np.zeros_like(islands, np.uint8), info={**base, **info})

    @property
    def size(self) -> tuple[int, int]:
        return self.islands.shape[1], self.islands.shape[0]

    @property
    def mask(self) -> np.ndarray:
        return self.islands > 0

    def island_ids(self) -> list[int]:
        return [i["id"] for i in self.info["islands"]]

    def island(self, k: int) -> dict[str, Any]:
        return next(i for i in self.info["islands"] if i["id"] == k)

    def resized(self, size: tuple[int, int]) -> "UVLayout":
        """Nearest-neighbour rescale of the maps (and seam/bbox coordinates) to another texture size."""
        if size == self.size:
            return self
        W, H = size
        sx, sy = W / self.size[0], H / self.size[1]
        ids = np.asarray(Image.fromarray(self.islands).resize(size, Image.Resampling.NEAREST), np.int32)
        ov = np.asarray(Image.fromarray(self.overlap).resize(size, Image.Resampling.NEAREST))
        ys = (np.arange(H) * self.size[1] / H).astype(int)
        xs = (np.arange(W) * self.size[0] / W).astype(int)
        nrm = self.normals[ys][:, xs] if self.normals is not None else None
        upa = self.up_angle[ys][:, xs] if self.up_angle is not None else None
        info = json.loads(json.dumps(self.info))
        info["size"] = [W, H]
        for i in info["islands"]:
            b = i["bbox"]
            i["bbox"] = [int(b[0] * sx), int(b[1] * sy), int(np.ceil(b[2] * sx)), int(np.ceil(b[3] * sy))]
        for s in info.get("seams", []):
            for side in ("a", "b"):
                s[side] = [[p[0] * sx, p[1] * sy] for p in s[side]]
        return UVLayout(ids, ov, info, nrm, upa)

    def boundaries(self) -> np.ndarray:
        """Texels of an island that touch a different island or empty space."""
        ids = self.islands
        edge = np.zeros(ids.shape, bool)
        edge[:, :-1] |= ids[:, :-1] != ids[:, 1:]
        edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
        edge[:-1, :] |= ids[:-1, :] != ids[1:, :]
        edge[1:, :] |= ids[1:, :] != ids[:-1, :]
        return edge & (ids > 0)

    def rotation_classes(self, mask: np.ndarray) -> dict[int, np.ndarray]:
        """Split ``mask`` by how the texture is rotated on the mesh: {k: texels whose world-up points
        k quarter-turns clockwise from image-up}. Rotating the atlas k quarter-turns counter-clockwise
        (``np.rot90(a, k)``) makes those texels upright."""
        if self.up_angle is None:
            return {0: mask}
        q = (np.round(np.nan_to_num(self.up_angle, nan=0.0) / 90).astype(int)) % 4
        return {k: mask & (q == k) for k in range(4) if (mask & (q == k)).any()}

    def orientation(self) -> Image.Image | None:
        """Which way each texel's surface faces, as colour: up = blue, down = dark olive,
        sideways = red/green by direction (world-space normal map). Tells the model roof from wall."""
        if self.normals is None:
            return None
        rgb = ((self.normals * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        rgb[~self.mask] = 0
        return Image.fromarray(rgb, "RGB")

    def preview(self) -> Image.Image:
        """Islands in distinct colours with numbered centroids — for humans and prompts."""
        rng = np.random.default_rng(7)
        palette = np.vstack([[40, 40, 40], rng.integers(70, 230, (int(self.islands.max()) + 1, 3))]).astype(np.uint8)
        img = Image.fromarray(palette[np.where(self.mask, self.islands + 1, 0)], "RGB")
        return label_islands(img, self)


def label_islands(img: Image.Image, uv: UVLayout, ids: Sequence[int] | None = None) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=max(12, min(uv.size) // 45))
    for k in ids or uv.island_ids():
        ys, xs = np.nonzero(uv.islands == k)
        if len(xs) == 0:
            continue
        # A texel inside the island nearest its centroid (centroids of L-shapes fall outside).
        cx, cy = xs.mean(), ys.mean()
        j = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        d.text((int(xs[j]), int(ys[j])), str(k), fill=(255, 255, 0), anchor="mm", font=font,
               stroke_width=max(1, min(uv.size) // 400), stroke_fill=(0, 0, 0))  # fmt: skip
    return out


def outline(img: Image.Image, uv: UVLayout, color=(255, 0, 0), width: int | None = None) -> Image.Image:
    """The atlas with every island boundary drawn on top (the layout guide for the model)."""
    width = width or max(1, min(uv.size) // 400)
    edge = uv.boundaries()
    if width > 1:
        edge = ndimage.binary_dilation(edge, iterations=width - 1) & uv.mask
    arr = np.asarray(img.convert("RGB")).copy()
    arr[edge] = color
    return Image.fromarray(arr, "RGB")


# --------------------------------------------------------------------------- #
# Pixel helpers
# --------------------------------------------------------------------------- #


def bleed(img: Image.Image, mask: np.ndarray, padding: int, keep: Image.Image | None = None) -> Image.Image:
    """Extend island colours ``padding`` px outward (nearest island texel), like Blender's bake margin.

    Texels farther than ``padding`` from any island are taken from ``keep`` (default: left as is).
    """
    arr = np.asarray(img.convert("RGB")).copy()
    if padding > 0 and mask.any() and not mask.all():
        dist, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
        ring = (~mask) & (dist <= padding)
        arr[ring] = arr[iy[ring], ix[ring]]
        far = (~mask) & (dist > padding)
    else:
        far = ~mask
    if keep is not None:
        k = np.asarray(keep.convert("RGB").resize(img.size))
        arr[far] = k[far]
    return Image.fromarray(arr, "RGB")


def _edges(img: Image.Image | np.ndarray, sigma: float = 1.0) -> np.ndarray:
    g = np.asarray(img.convert("L") if isinstance(img, Image.Image) else img, np.float64)
    g = ndimage.gaussian_filter(g, sigma)
    return np.hypot(ndimage.sobel(g, 1), ndimage.sobel(g, 0))


def _sample(arr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear samples of an [H, W, C] array at (x, y) pixel coords (pixel centres at +0.5)."""
    H, W = arr.shape[:2]
    x = np.clip(pts[:, 0] - 0.5, 0, W - 1)
    y = np.clip(pts[:, 1] - 0.5, 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x1] * fx * (1 - fy)
            + arr[y1, x0] * (1 - fx) * fy + arr[y1, x1] * fx * fy)  # fmt: skip


def _seam_points(seam: dict, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    a0, a1 = np.array(seam["a"], float)
    b0, b1 = np.array(seam["b"], float)
    n = max(2, int(max(np.linalg.norm(a1 - a0), np.linalg.norm(b1 - b0)) / step))
    t = ((np.arange(n) + 0.5) / n)[:, None]
    return a0 + t * (a1 - a0), b0 + t * (b1 - b0)


def placeholder(uv: UVLayout, base: ImageLike | None = None) -> Image.Image:
    """Flat colour per island by which way it faces — a stand-in for an untextured atlas."""
    W, H = uv.size
    arr = np.asarray(ops.load_image(base).convert("RGB").resize((W, H))) if base is not None else np.full((H, W, 3), 60, np.uint8)
    arr = arr.copy()
    roof, under, wall = (150, 70, 55), (95, 95, 100), (205, 190, 160)
    if uv.normals is not None:  # per texel, so an island holding a wall and a roof gets both
        nz = uv.normals[..., 2]
        arr[uv.mask & (nz > 0.35)] = roof
        arr[uv.mask & (nz < -0.35)] = under
        arr[uv.mask & (np.abs(nz) <= 0.35)] = wall
        return Image.fromarray(arr, "RGB")
    for isl in uv.info["islands"]:
        n = isl.get("normal") or [0, 0, 0]
        arr[uv.islands == isl["id"]] = roof if n[2] > 0.35 else under if n[2] < -0.35 else wall
    return Image.fromarray(arr, "RGB")


# --------------------------------------------------------------------------- #
# Retexture
# --------------------------------------------------------------------------- #

_RETEXTURE = """\
Image 1 is a UV texture atlas: the unwrapped surface texture of a 3D model. Every coloured
region ("UV island") maps onto part of the model's surface. Image 2 is the same atlas with each
island's outline drawn in red{labels}.{orientation}
{style_refs} show the look the model should have.

Repaint image 1 so the model's surfaces get the materials, colours and surface detail shown in the
reference{s}.{instruction}

This is a texture map, not a picture, so:
- Keep every island exactly where it is, with the same shape, size and orientation, pixel-aligned
  with image 1. Do not move, rescale, rotate, re-crop or re-arrange anything.
- Paint inside the islands only, and fill each island completely right up to its outline. Leave the
  empty background between islands as it is.
- Keep any features already painted inside the islands (markings, seams, eyes, panels) in the same
  positions.
- Paint flat albedo colour: no lighting, shading, shadows, highlights, reflections or perspective.
  Use the same pattern scale on every island.
- No text, labels, numbers, outlines or red lines in the output.
Output only the repainted atlas, at the same aspect ratio as image 1."""

_ORIENTATION = """
Image 3 shows which way the surface under each texel faces on the 3D model: blue = faces up
(roofs, tops, floors seen from above), dark olive = faces down (undersides), red/green/pink/teal =
faces sideways (walls, sides). One island can contain several surfaces, e.g. a wall and a roof; use
image 3 to put the right material on each part."""


@dataclass
class RetextureResult:
    image: Image.Image  # final atlas: model output clamped to islands + bleed
    raw: Image.Image  # the model's unclamped output at atlas size (what the drift check inspects)
    guide: Image.Image
    calls: list[dict[str, Any]] = field(default_factory=list)
    groups: list[dict[str, Any]] = field(default_factory=list)  # materials mode: what was painted where


def _island_notes_text(notes: dict[int, str] | None) -> str:
    if not notes:
        return ""
    lines = "\n".join(f"- island {k}: {v}" for k, v in sorted(notes.items()))
    return f"\nPer-island materials (numbers are the yellow labels in image 2):\n{lines}"


# --------------------------------------------------------------------------- #
# Material groups: which texels get which material
# --------------------------------------------------------------------------- #

SELECTORS = "up | down | side | +x | -x | +y | -y | +z | -z | islands:1,2,5 | all | rest"


def select_texels(uv: UVLayout, selector: str, taken: np.ndarray | None = None, up: float = 0.35) -> np.ndarray:
    """Texel mask for a selector, using Blender's per-texel world normals (up = +Z).

    ``up``/``down``/``side`` split by how steeply the surface faces up (normal.z beyond ±``up``);
    ``+x`` etc. pick the dominant facing axis; ``islands:1,3`` picks UV islands; ``rest`` is every
    island texel not taken by earlier selectors.
    """
    sel = selector.strip().lower()
    mask = uv.mask
    if sel == "all":
        return mask.copy()
    if sel == "rest":
        return mask & ~(taken if taken is not None else np.zeros_like(mask))
    if sel.startswith("islands:"):
        ids = [int(k) for k in sel.split(":", 1)[1].split(",") if k.strip()]
        return np.isin(uv.islands, ids)
    if uv.normals is None:
        raise ValueError(f"Selector {selector!r} needs normals; re-run `imagegen uv export` (older exports lack them)")
    n = uv.normals
    if sel == "up":
        return mask & (n[..., 2] > up)
    if sel == "down":
        return mask & (n[..., 2] < -up)
    if sel == "side":
        return mask & (np.abs(n[..., 2]) <= up)
    axes = {"x": 0, "y": 1, "z": 2}
    if len(sel) == 2 and sel[0] in "+-" and sel[1] in axes:
        dom = np.argmax(np.abs(n), axis=-1) == axes[sel[1]]
        sign = n[..., axes[sel[1]]] > 0 if sel[0] == "+" else n[..., axes[sel[1]]] < 0
        return mask & dom & sign
    raise ValueError(f"Unknown selector {selector!r}; use {SELECTORS}")


def material_groups(uv: UVLayout, materials: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    """[(selector, description)] -> non-overlapping groups, in order (earlier selectors win)."""
    taken = np.zeros(uv.mask.shape, bool)
    groups = []
    for selector, desc in materials:
        m = select_texels(uv, selector, taken) & ~taken
        taken |= m
        groups.append({"selector": selector, "material": desc, "mask": m, "texels": int(m.sum())})
    left = uv.mask & ~taken
    if left.sum() > 0.002 * uv.mask.sum():
        groups.append({"selector": "(unassigned)", "material": None, "mask": left, "texels": int(left.sum())})
    return groups


def parse_materials(items: Sequence[str] | None) -> list[tuple[str, str]]:
    """``["up=terracotta roof tiles", "side=fieldstone"]`` -> [("up", "terracotta roof tiles"), ...]."""
    out = []
    for it in items or []:
        sel, _, desc = it.partition("=")
        if not desc.strip():
            raise ValueError(f"Bad material {it!r}; expected SELECTOR=description ({SELECTORS})")
        out.append((sel.strip(), desc.strip()))
    return out


_GROUP = """\
Image 1 is (part of) a UV texture atlas: the unwrapped surface texture of a 3D model, turned so that
"up" on the model points up in the image. The area marked in image 2 is every part of the model that
should be: {material}.
Paint the marked area as {material}, matching the look (colours, material, surface detail, wear) of
the reference image{s}. Fill the marked area completely right up to its edges. Keep the pattern at one
consistent scale everywhere, oriented the way it would be on the real object (e.g. brick courses and
tile rows horizontal, planks and siding the way they run){direction}. Paint flat albedo colour: no lighting, shading, shadows, highlights
or perspective. No text, labels or outlines.{instruction}"""


def _paint_mask(
    img: Image.Image,
    mask: np.ndarray,
    prompt: str,
    styles: Sequence[Image.Image],
    backend: Backend,
    context: float = 0.25,
    rotate: int = 0,
) -> tuple[Image.Image, Image.Image, np.ndarray, list[dict[str, Any]]]:
    """One masked edit, done on the atlas turned ``rotate`` quarter-turns counter-clockwise so the
    painted surface is upright for the model, then turned back (lossless).

    Returns (image with only ``mask`` texels replaced, the model's unclamped patch pasted over the
    image, a mask of where that patch covers, call info).
    """
    arr = np.asarray(img.convert("RGB"))
    a_r, m_r = np.rot90(arr, rotate), np.rot90(mask, rotate)
    grown = ndimage.binary_dilation(m_r, iterations=3)  # paint past the edge; the clamp trims it
    m_img = Image.fromarray((grown * 255).astype(np.uint8), "L")
    res = pipeline.edit(Image.fromarray(np.ascontiguousarray(a_r), "RGB"), prompt, backend=backend, mask=m_img,
                        context=context, feather=0, references=list(styles), color_match=False)  # fmt: skip
    l, t, r, b = res.context_box
    patch = np.asarray(ops.fit(res.generations[0].image.convert("RGB"), (r - l, b - t), "stretch"))
    crop, crop_mask = a_r[t:b, l:r], grown[t:b, l:r]
    # Image models often shift/offset what they paint by a few percent. Left alone, the texels the
    # shift uncovers keep whatever was there before (background, a neighbouring material): a strip
    # along one edge of the island. Find where the *changed* region sits relative to the mask and
    # translate the patch back before compositing.
    dx, dy, gain = register_patch(crop, patch, crop_mask)
    if dx or dy:
        patch = ndimage.shift(patch, (-dy, -dx, 0), order=0, mode="nearest")
    patch_full = a_r.copy()
    patch_full[t:b, l:r] = patch
    covered = np.zeros(m_r.shape, bool)
    covered[t:b, l:r] = True
    painted = patch_full  # feather=0 and no colour match: the edit is exactly the patch inside the mask
    edited = np.rot90(painted, -rotate)
    out = arr.copy()
    out[mask] = edited[mask]
    calls = res.info()["calls"]
    for c in calls:
        c["shift_corrected"] = [int(dx), int(dy)]
        c["shift_gain"] = round(gain, 3)
    return (Image.fromarray(out, "RGB"), Image.fromarray(np.ascontiguousarray(np.rot90(patch_full, -rotate)), "RGB"),
            np.rot90(covered, -rotate), calls)  # fmt: skip


def register_patch(before: np.ndarray, after: np.ndarray, mask: np.ndarray, search_frac: float = 0.12,
                   min_gain: float = 1.25) -> tuple[int, int, float]:  # fmt: skip
    """How far the model displaced its painting: the (dx, dy) that best lines the changed-texel map
    up with ``mask``. Returns (0, 0, gain) unless moving beats staying put by ``min_gain``.

    Score(shift) = correlation between the change map and the shifted, slightly blurred mask. The
    blur makes the score peak at the *centred* alignment when the model painted a bit wider or
    narrower than the mask. Searched coarse (1/4 resolution), then refined at full resolution.
    """
    D = np.abs(after.astype(np.float32) - before.astype(np.float32)).mean(axis=2)
    D = ndimage.gaussian_filter(D, 1.5)
    H, W = D.shape
    M = ndimage.gaussian_filter(mask.astype(np.float32), max(2.0, 0.01 * max(H, W)))
    if mask.sum() < 50 or (~mask).sum() < 50 or D.std() < 1e-3:
        return 0, 0, 1.0

    def score(Dm: np.ndarray, Mm: np.ndarray, sx: int, sy: int) -> float:
        sh = np.roll(Mm, (sy, sx), axis=(0, 1))
        if sy > 0: sh[:sy] = 0  # noqa: E701 — blank wrapped-in rows/cols
        elif sy < 0: sh[sy:] = 0  # noqa: E701
        if sx > 0: sh[:, :sx] = 0  # noqa: E701
        elif sx < 0: sh[:, sx:] = 0  # noqa: E701
        if sh.std() < 1e-6:
            return -1.0
        return float(np.corrcoef(Dm.ravel(), sh.ravel())[0, 1])

    search = max(12, int(search_frac * max(H, W)))
    f = 8 if min(H, W) >= 512 else 4 if min(H, W) >= 128 else 1
    Dc, Mc = D[::f, ::f], M[::f, ::f]
    sc = max(1, search // f)
    best, bx, by = -2.0, 0, 0
    for sy in range(-sc, sc + 1):
        for sx in range(-sc, sc + 1):
            v = score(Dc, Mc, sx, sy)
            if v > best:
                best, bx, by = v, sx, sy
    best, fx, fy = -2.0, 0, 0
    for sy in range(by * f - f, by * f + f + 1):
        for sx in range(bx * f - f, bx * f + f + 1):
            v = score(D, M, sx, sy)
            if v > best:
                best, fx, fy = v, sx, sy
    zero = score(D, M, 0, 0)
    # How much of the misalignment the shift removes: (1 - corr at 0) / (1 - corr at best).
    gain = (1 - zero) / max(1 - best, 1e-6)
    if (fx, fy) == (0, 0) or gain < min_gain:
        return 0, 0, float(min(gain, 99.0))
    return fx, fy, float(min(gain, 99.0))


def drift_regions(uv: "UVLayout", materials: Sequence[tuple[str, str]] | None = None,
                  min_texels: int | None = None) -> list[tuple[int, str | None, np.ndarray]]:  # fmt: skip
    """Regions whose outline the model's paint should follow: every island, or with materials every
    (island x material) part — a pass can drift against an island's *internal* wall/roof border."""
    W, H = uv.size
    floor = min_texels if min_texels is not None else max(400, 0.0005 * W * H)
    groups = [g for g in material_groups(uv, materials) if g["material"]] if materials else []
    out = []
    for isl in uv.info["islands"]:
        k = isl["id"]
        m = uv.islands == k
        parts = [(g["selector"], g["mask"] & m) for g in groups if (g["mask"] & m).sum() >= floor] if groups else []
        if len(parts) > 1:
            out.extend((k, sel, pm) for sel, pm in parts)
        elif m.sum() >= floor:
            out.append((k, None, m))
    return out


def measure_drift(raw: ImageLike, uv: "UVLayout", materials: Sequence[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    """Per-region offset of the model's painting vs the true outline (see ``_region_drift``)."""
    raw_img = ops.load_image(raw).convert("RGB").resize(uv.size)
    W, H = uv.size
    edge_raw = _edges(raw_img)
    search = int(max(6, round(0.02 * max(W, H))))
    out = [_region_drift(edge_raw, m, k, search, sel) for k, sel, m in drift_regions(uv, materials)]
    return [d for d in out if "skipped" not in d]


def offset_strips(uv: "UVLayout", drift: Sequence[dict[str, Any]], new: ImageLike, original: ImageLike,
                  max_shift: float = 2.0, min_prominence: float = 1.25,
                  materials: Sequence[tuple[str, str]] | None = None) -> np.ndarray:  # fmt: skip
    """Texels an offset left unpainted.

    If a pass painted island k at (island + shift), the texels of the island outside that shifted
    copy weren't painted — the model reproduced what it was shown there, displaced by the same
    shift (background, the old flat colour, a neighbouring material). So a texel p in that band is
    leftover when it still looks like ``original[p - shift]``; once filled or repainted it doesn't.
    """
    n = np.asarray(ops.load_image(new).convert("RGB").resize(uv.size), np.float32)
    o = np.asarray(ops.load_image(original).convert("RGB").resize(uv.size), np.float32)
    strips = np.zeros(uv.islands.shape, bool)
    regions = {(k, sel): m for k, sel, m in drift_regions(uv, materials)}

    def local_std(a: np.ndarray) -> np.ndarray:
        g = a.mean(axis=2)
        return np.sqrt(np.maximum(ndimage.uniform_filter(g**2, 5) - ndimage.uniform_filter(g, 5) ** 2, 0))

    ls_n = local_std(n)
    for d in drift:
        dx, dy = d["best_shift"]
        if np.hypot(dx, dy) <= max_shift or d["peak_prominence"] <= min_prominence:
            continue
        isl = regions.get((d["island"], d.get("region")))
        if isl is None:
            isl = uv.islands == d["island"]
        moved = ndimage.shift(isl.astype(np.uint8), (dy, dx), order=0, cval=0).astype(bool)
        band = isl & ~moved
        if not band.any():
            continue
        o_shifted = ndimage.shift(o, (dy, dx, 0), order=0, mode="nearest")  # original content moved by the offset
        # a copy matches in colour *and* texture (a dark textured stone isn't a copy of flat background)
        copied = (ndimage.uniform_filter(np.abs(n - o_shifted).max(axis=2), 3) < 14) & (np.abs(ls_n - local_std(o_shifted)) < 4)
        strips |= band & copied
    return strips


def fill_holes(atlas: ImageLike, holes: np.ndarray, uv: "UVLayout") -> Image.Image:
    """Fill ``holes`` (island texels the model left unpainted) by mirroring nearby painted texels of
    the same island across the hole's edge — keeps texture instead of smearing one colour."""
    img = np.asarray(ops.load_image(atlas).convert("RGB")).copy()
    H, W = holes.shape
    for k in np.unique(uv.islands[holes]):
        if not k:
            continue
        isl = uv.islands == k
        # Grow the hole 2 px: the texels bordering a leftover strip are usually a blend of the strip
        # and the paint (anti-aliasing), which shows as a thin line after filling.
        todo = ndimage.binary_dilation(holes & isl, iterations=2) & isl
        good = isl & ~todo
        if not good.any() or not todo.any():
            continue
        _, (iy, ix) = ndimage.distance_transform_edt(~good, return_indices=True)
        ys, xs = np.nonzero(todo)
        qy, qx = iy[ys, xs], ix[ys, xs]  # nearest painted texel
        ry, rx = np.clip(2 * qy - ys, 0, H - 1), np.clip(2 * qx - xs, 0, W - 1)  # reflected across the edge
        ok = good[ry, rx]
        img[ys, xs] = np.where(ok[:, None], img[ry, rx], img[qy, qx])
    return Image.fromarray(img, "RGB")


def unpainted(new: ImageLike, original: ImageLike, uv: "UVLayout",
              materials: Sequence[tuple[str, str]] | None = None) -> np.ndarray:  # fmt: skip
    """Island texels that were meant to be repainted but weren't: transparent, flat background
    colour, or still (locally) identical to the original atlas — e.g. a strip of the old flat colour
    or of a neighbouring material left behind when the model's paint was offset."""
    n_img, o_img = ops.load_image(new), ops.load_image(original)
    uv = uv.resized(o_img.size)
    n = np.asarray(n_img.convert("RGB").resize(o_img.size), np.float32)
    o = np.asarray(o_img.convert("RGB"), np.float32)
    mask = uv.mask
    target = mask
    if materials:
        target = np.zeros_like(mask)
        for g in material_groups(uv, materials):
            if g["material"]:
                target |= g["mask"]
    holes = np.zeros(mask.shape, bool)
    if n_img.mode == "RGBA":
        holes |= np.asarray(n_img.getchannel("A").resize(o_img.size)) < 250

    # Neighbourhood statistics over island texels only, so padding/background next to an island
    # edge doesn't dilute them (strips of unpainted texels sit exactly on island edges).
    w = ndimage.uniform_filter(mask.astype(np.float32), 5)

    def mmean(a: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(a * mask, 5) / np.maximum(w, 1e-6)

    def local_std(a: np.ndarray) -> np.ndarray:
        g = a.mean(axis=2)
        return np.sqrt(np.maximum(mmean(g**2) - mmean(g) ** 2, 0))

    ls_n = local_std(n)
    if (~mask).any():
        bg = np.median(o[~mask], axis=0)
        holes |= (np.abs(n - bg).max(axis=2) < 12) & (ls_n < 3) & ~(np.abs(o - bg).max(axis=2) < 12)
    # unchanged from the original, as a neighbourhood (a repainted texture differs almost everywhere)
    same = mmean(np.abs(n - o).max(axis=2)) < 6
    holes |= same & (np.abs(ls_n - local_std(o)) < 3)
    # flat, texture-less patches inside a region whose material is otherwise textured: something
    # the model didn't paint (or painted as a plain fill), whatever its colour
    segs = [g["mask"] & (uv.islands == k) for g in material_groups(uv, materials) if g["material"]
            for k in uv.island_ids()] if materials else [uv.islands == k for k in uv.island_ids()]  # fmt: skip
    for seg in segs:
        if seg.sum() < 200:
            continue
        med = float(np.median(ls_n[seg]))
        if med < 8:  # a genuinely flat material: nothing to compare against
            continue
        flat = seg & (ls_n < max(2.0, 0.12 * med))
        lab, nlab = ndimage.label(flat)
        if nlab:
            sizes = ndimage.sum(flat, lab, index=np.arange(1, nlab + 1))
            keep = np.flatnonzero(np.asarray(sizes) >= max(40, 0.002 * seg.sum())) + 1
            holes |= np.isin(lab, keep)
    return holes & target


def _upright_crop(img: Image.Image, mask: np.ndarray, rotate: int) -> Image.Image:
    """The painted texels of ``mask``, turned upright and cropped (neutral grey elsewhere)."""
    a = np.rot90(np.asarray(img.convert("RGB")), rotate)
    m = np.rot90(mask, rotate)
    ys, xs = np.nonzero(m)
    t, b, l, r = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.where(m[t:b, l:r, None], a[t:b, l:r], 128).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(crop), "RGB")


def _texel_density(uv: UVLayout, mask: np.ndarray) -> float | None:
    dens = {i["id"]: i.get("texel_density") for i in uv.info["islands"]}
    ids, counts = np.unique(uv.islands[mask], return_counts=True)
    pairs = [(dens.get(int(k)), c) for k, c in zip(ids, counts) if k and dens.get(int(k))]
    return sum(d * c for d, c in pairs) / sum(c for _, c in pairs) if pairs else None


def _scaled_anchor(anchor: Image.Image, uv: UVLayout, anchor_mask: np.ndarray, target_mask: np.ndarray) -> Image.Image:
    """Rescale the anchor so one metre on the model is as many pixels as in the target region."""
    da, dt = _texel_density(uv, anchor_mask), _texel_density(uv, target_mask)
    if not da or not dt or abs(dt / da - 1) < 0.05:
        return anchor
    f = dt / da
    return anchor.resize((max(8, round(anchor.width * f)), max(8, round(anchor.height * f))), Image.Resampling.LANCZOS)


def _paste_raw(raw: Image.Image, patch_full: Image.Image, covered: np.ndarray, region: np.ndarray, grow: int = 12) -> Image.Image:
    """Put a model patch into the running raw image around ``region`` (so its edges near the
    region's island boundaries are what the drift check sees)."""
    arr = np.asarray(raw.convert("RGB")).copy()
    near = ndimage.binary_dilation(region, iterations=grow) & covered
    arr[near] = np.asarray(patch_full)[near]
    return Image.fromarray(arr, "RGB")


# --------------------------------------------------------------------------- #
# Retexture / repair
# --------------------------------------------------------------------------- #


def retexture(
    atlas: ImageLike,
    uv: UVLayout,
    styles: Sequence[ImageLike],
    *,
    backend: Backend,
    instruction: str = "",
    island_notes: dict[int, str] | None = None,
    materials: Sequence[tuple[str, str]] | None = None,
    padding: int = 16,
    mode: str = "auto",
    tile: int = 1024,
    use_orientation: bool = True,
    concurrency: int = 4,
    log: Log = _nolog,
) -> RetextureResult:
    """Repaint ``atlas`` in the look of ``styles`` while keeping the UV layout registered.

    mode:
      ``materials`` — one masked edit per material group (``materials=[("up", "roof tiles"),
                      ("side", "stone wall")]``), run in parallel. Most reliable when the object
                      has distinct surfaces: the model gets one unambiguous job per call.
      ``whole``     — one call for the whole atlas (best cross-island consistency).
      ``tiles``     — overlapping tiles (more detail on big atlases, less consistency).
      ``islands``   — one masked edit per UV island (slow; tight per-island control).
      ``auto``      — materials if given, else whole up to 2048 px, else tiles.
    """
    base = ops.load_image(atlas).convert("RGB")
    uv = uv.resized(base.size)
    style_imgs = [ops.load_image(s).convert("RGB") for s in styles]
    if not style_imgs:
        raise ValueError("Need at least one style reference image")
    guide = outline(base, uv)
    if island_notes:
        guide = label_islands(guide, uv, list(island_notes))
    if mode == "auto":
        mode = "materials" if materials else ("whole" if max(base.size) <= 2048 else "tiles")
    calls: list[dict[str, Any]] = []
    extra = f"\n{instruction.strip()}" if instruction.strip() else ""
    s = "" if len(style_imgs) == 1 else "s"

    if mode == "materials":
        if not materials:
            raise ValueError("mode='materials' needs materials=[(selector, description), ...]")
        groups = [g for g in material_groups(uv, materials) if g["texels"]]
        unassigned = [g for g in groups if g["material"] is None]
        if unassigned:
            log(f"warning: {unassigned[0]['texels']} island texels match no material selector; they keep the "
                "original texture (add a 'rest=...' material to cover them)")  # fmt: skip
        # One job per (material, UV rotation): each is painted on an atlas turned upright for it.
        jobs = [(g, k, sub) for g in groups if g["material"]
                for k, sub in uv.rotation_classes(g["mask"]).items() if sub.sum() >= 50]  # fmt: skip
        log(f"retexture: {len(jobs)} masked pass(es): " + ", ".join(
            f"{g['selector']}={g['material']!r}@{k * 90}deg ({int(sub.sum())} texels)" for g, k, sub in jobs))

        # Passes of the same material are separate model calls, so they drift apart in scale and
        # colour. Paint the largest pass of each material first; the others then get that result
        # (upright, rescaled by texel density) as an extra reference to match.
        jobs.sort(key=lambda j: -int(j[2].sum()))
        firsts: dict[str, tuple] = {}
        for j in jobs:
            firsts.setdefault(j[0]["material"], j)
        rest = [j for j in jobs if firsts[j[0]["material"]] is not j]
        anchors: dict[str, Image.Image] = {}

        def work(job):
            g, k, sub = job
            refs, note = list(style_imgs), ""
            anchor = anchors.get(g["material"]) if any(job is r for r in rest) else None
            if anchor is not None:
                refs.append(_scaled_anchor(anchor, uv, firsts[g["material"]][2], sub))
                note = ("\nThe last reference image shows this same material already painted on another part of this "
                        "model: match its pattern size, colours and detail exactly so the parts look like one surface.")  # fmt: skip
            prompt = _GROUP.format(material=g["material"], s=s, instruction=extra + note, direction="")
            # A little context: the model sees the boundary it must respect, and registration has
            # untouched texels around the mask to measure an offset against.
            return sub, _paint_mask(base, sub, prompt, refs, backend, context=0.1, rotate=k)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            first_results = list(pool.map(work, firsts.values()))
            for (g, k, sub), (_, (img, *_r)) in zip(firsts.values(), first_results):
                anchors[g["material"]] = _upright_crop(img, sub, k)
            if rest:
                log(f"retexture: matching {len(rest)} more pass(es) to the first pass of their material")
            results = first_results + list(pool.map(work, rest))
        painted = np.asarray(base).copy()
        raw = base.copy()
        for sub, (img, patch_full, covered, info) in results:
            painted[sub] = np.asarray(img)[sub]
            raw = _paste_raw(raw, patch_full, covered, sub)
            calls.extend(info)
        final = clamp_to_islands(Image.fromarray(painted, "RGB"), base, uv, padding)
        groups_info = [{k: v for k, v in g.items() if k != "mask"} for g in groups]
        return RetextureResult(image=final, raw=raw, guide=guide, calls=calls, groups=groups_info)

    orient = uv.orientation() if use_orientation else None
    guides = [guide] + ([orient] if orient is not None else [])
    n, first = len(style_imgs), 1 + len(guides)  # images: atlas, guides..., styles...
    prompt = _RETEXTURE.format(
        labels=" and the islands that have instructions numbered in yellow" if island_notes else "",
        orientation=_ORIENTATION if orient is not None else "",
        style_refs=f"Image {first + 1}" if n == 1 else f"Images {first + 1}-{first + n}",
        s=s,
        instruction=extra + _island_notes_text(island_notes),
    )

    if mode in ("whole", "tiles") and len(uv.rotation_classes(uv.mask)) > 1:
        log("warning: UV islands are rotated differently relative to the mesh; a directional pattern painted in "
            "one pass will run different ways on the model. Use --material (materials mode) to paint upright.")  # fmt: skip
    if mode == "whole":
        log(f"retexture: one call for the whole {base.size[0]}x{base.size[1]} atlas")
        res = backend.generate(prompt, references=[base, *guides, *style_imgs],
                               aspect_ratio=ops.nearest_aspect(*base.size, backend.aspect_ratios))  # fmt: skip
        raw = ops.fit(res.image.convert("RGB"), base.size, "stretch")
        calls.append({"backend": res.backend, "model": res.model, **res.meta})
    elif mode == "tiles":
        from concurrent.futures import ThreadPoolExecutor

        boxes = [b for b in ops.tile_boxes(base.size, tile, tile // 8)
                 if uv.mask[b[1]:b[3], b[0]:b[2]].any()]  # fmt: skip
        log(f"retexture: {len(boxes)} tile(s) of {tile}px")

        def tile_work(box):
            r = backend.generate(prompt, references=[base.crop(box), *(g.crop(box) for g in guides), *style_imgs],
                                 aspect_ratio=ops.nearest_aspect(box[2] - box[0], box[3] - box[1], backend.aspect_ratios))  # fmt: skip
            calls.append({"backend": r.backend, "model": r.model, "tile": list(box), **r.meta})
            return box, ops.fit(r.image.convert("RGB"), (box[2] - box[0], box[3] - box[1]), "stretch")

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            tiles = list(pool.map(tile_work, boxes))
        blended = ops.blend_tiles(base.size, tiles, "RGB")
        covered = np.zeros(uv.mask.shape, bool)
        for (l, t, r, b), _ in tiles:
            covered[t:b, l:r] = True
        raw = Image.fromarray(np.where(covered[..., None], np.asarray(blended), np.asarray(base)), "RGB")
    elif mode == "islands":
        painted, raw = base.copy(), base.copy()
        for k in uv.island_ids():
            note = (island_notes or {}).get(k, "")
            log(f"retexture: island {k}")
            painted, raw, info = _repaint_island(painted, raw, uv, k, style_imgs, backend, instruction, note)
            calls.extend(info)
        return RetextureResult(image=clamp_to_islands(painted, base, uv, padding), raw=raw, guide=guide, calls=calls)
    else:
        raise ValueError("mode must be auto, materials, whole, tiles or islands")

    return RetextureResult(image=clamp_to_islands(raw, base, uv, padding), raw=raw, guide=guide, calls=calls)


def clamp_to_islands(raw: Image.Image, original: Image.Image, uv: UVLayout, padding: int) -> Image.Image:
    """Take island texels from ``raw``, everything else from ``original``, then bleed ``padding`` px."""
    r = np.asarray(raw.convert("RGB").resize(uv.size))
    o = np.asarray(original.convert("RGB").resize(uv.size))
    merged = Image.fromarray(np.where(uv.mask[..., None], r, o), "RGB")
    return bleed(merged, uv.mask, padding, keep=original)


_ISLAND = """\
Image 1 is part of a UV texture atlas (the unwrapped texture of a 3D model); the area marked in
image 2 is one UV island. Repaint that island with the material, colours and surface detail of the
reference image(s). Fill the marked area completely, keep its outline and any features inside it
exactly where they are, and paint flat albedo colour: no lighting, shadows or perspective, no text.
{instruction}"""


def _repaint_island(painted: Image.Image, raw: Image.Image, uv: UVLayout, k: int, styles: Sequence[Image.Image],
                    backend: Backend, instruction: str = "", note: str = "",
                    materials: Sequence[tuple[str, str]] | None = None):  # fmt: skip
    """Repaint island k. With ``materials``, each material group inside the island is painted by
    its own call (so a wall+roof island keeps both); otherwise one call for the whole island."""
    m = uv.islands == k
    jobs: list[tuple[np.ndarray, str]] = []
    if materials:
        for g in material_groups(uv, materials):
            sub = g["mask"] & m
            if g["material"] and sub.sum() > 50:
                jobs.append((sub, _GROUP.format(material=g["material"], s="(s)", direction="",
                                                instruction=f"\n{instruction.strip()}" if instruction.strip() else "")))  # fmt: skip
    if not jobs:
        extra = " ".join(x for x in (instruction.strip(), f"This island: {note}." if note else "") if x)
        jobs.append((m, _ISLAND.format(instruction=extra)))
    calls: list[dict[str, Any]] = []
    for sub, prompt in jobs:
        for rot, part in uv.rotation_classes(sub).items():
            if part.sum() < 50:
                continue
            new, patch_full, covered, info = _paint_mask(painted, part, prompt, styles, backend, rotate=rot)
            painted = new
            raw = _paste_raw(raw, patch_full, covered, part)
            calls.extend(info)
    return painted, raw, calls


def repair(
    atlas: ImageLike,
    original: ImageLike,
    uv: UVLayout,
    islands: Sequence[int],
    styles: Sequence[ImageLike],
    *,
    backend: Backend,
    raw: ImageLike | None = None,
    instruction: str = "",
    island_notes: dict[int, str] | None = None,
    materials: Sequence[tuple[str, str]] | None = None,
    padding: int = 16,
    log: Log = _nolog,
) -> tuple[Image.Image, Image.Image]:
    """Repaint only ``islands`` (e.g. a check's ``repair_islands``) with masked edits.

    Returns (final, raw): pass both back to ``check`` — ``raw`` carries the new patches unclamped,
    so drift in the repair itself is caught too.
    """
    img = ops.load_image(atlas).convert("RGB")
    orig = ops.load_image(original).convert("RGB")
    uv = uv.resized(img.size)
    # Start from un-bled texels: repaint on island content, never on the padding margin.
    painted = Image.fromarray(np.where(uv.mask[..., None], np.asarray(img), np.asarray(orig)), "RGB")
    raw_img = ops.load_image(raw).convert("RGB").resize(img.size) if raw is not None else painted.copy()
    style_imgs = [ops.load_image(s).convert("RGB") for s in styles]
    for k in islands:
        log(f"repair: island {k}")
        painted, raw_img, _ = _repaint_island(painted, raw_img, uv, k, style_imgs, backend, instruction,
                                              (island_notes or {}).get(k, ""), materials)  # fmt: skip
    return clamp_to_islands(painted, orig, uv, padding), raw_img


def fix_seams(atlas: ImageLike, uv: UVLayout, width: int = 6, strength: float = 1.0,
              original: ImageLike | None = None) -> Image.Image:  # fmt: skip
    """Pull both sides of every UV seam toward their shared average tone.

    Texture *patterns* can't be made continuous in 2D, but tone mismatch across a seam (the
    most visible artefact) can: the correction is spread ``width`` px into each island and fades out.
    With ``original``, only seam points that were continuous there are touched, so intended
    material boundaries (roof edge meets wall) stay crisp.
    """
    img = ops.load_image(atlas).convert("RGB")
    uv = uv.resized(img.size)
    arr = np.asarray(img, np.float64)
    smooth = ndimage.gaussian_filter(arr, (2, 2, 0))
    orig = ndimage.gaussian_filter(np.asarray(ops.load_image(original).convert("RGB").resize(img.size), np.float64), (2, 2, 0)) \
        if original is not None else None  # fmt: skip
    H, W = uv.islands.shape
    delta = np.zeros_like(arr)
    weight = np.zeros((H, W))
    for s in uv.info.get("seams", []):
        pa, pb = _seam_points(s)
        if orig is not None:
            keep = np.abs(_sample(orig, pa) - _sample(orig, pb)).mean(axis=1) < 15
            pa, pb = pa[keep], pb[keep]
            if not len(pa):
                continue
        ca, cb = _sample(smooth, pa), _sample(smooth, pb)
        avg = (ca + cb) / 2
        for pts, c in ((pa, ca), (pb, cb)):
            xi = np.clip(pts[:, 0].astype(int), 0, W - 1)
            yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
            np.add.at(delta, (yi, xi), (avg - c) * strength)
            np.add.at(weight, (yi, xi), 1.0)
    sigma = max(1.0, width / 2)
    spread_d = ndimage.gaussian_filter(delta, (sigma, sigma, 0))
    spread_w = ndimage.gaussian_filter(weight, sigma)
    corr = spread_d / np.maximum(spread_w, 1e-9)[..., None]
    fade = np.clip(spread_w / (spread_w.max() + 1e-9) * 4, 0, 1)[..., None]
    out = np.where(uv.mask[..., None], arr + corr * fade, arr)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def finish(
    atlas: ImageLike,
    original: ImageLike,
    uv: UVLayout,
    *,
    materials: Sequence[tuple[str, str]] | None = None,
    padding: int = 16,
    fill: bool = True,
    max_fill: float = 0.05,
    seams: bool = True,
    flatten: bool = False,
    raw: ImageLike | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Deterministic clean-up after any model pass: (flatten lighting), fill small unpainted strips,
    blend tone across seams that were continuous in the original, re-bleed the padding.

    Only islands with at most ``max_fill`` unpainted texels are filled; bigger gaps are left for
    ``check`` to report and ``repair`` to repaint, since mirroring can't invent that much texture.
    """
    orig = ops.load_image(original).convert("RGB")
    out = ops.load_image(atlas).convert("RGB")
    uvr = uv.resized(orig.size)
    info: dict[str, Any] = {"filled_islands": {}}
    if flatten:
        masks = [g["mask"] for g in material_groups(uvr, materials)] if materials else None
        out = flatten_lighting(out, uvr, masks)
    if fill:
        holes = unpainted(out, orig, uvr, materials)
        if raw is not None:  # strips uncovered by paint the offset corrector didn't catch
            holes |= offset_strips(uvr, measure_drift(raw, uvr, materials), out, orig, max(2.0, 0.004 * max(orig.size)),
                                   materials=materials)  # fmt: skip
        small = np.zeros_like(holes)
        for k in np.unique(uvr.islands[holes]):
            isl = uvr.islands == k
            frac = float((holes & isl).sum() / isl.sum()) if k else 0
            if k and frac <= max_fill:
                small |= holes & isl
                info["filled_islands"][int(k)] = round(frac, 4)
        if small.any():
            out = fill_holes(out, small, uvr)
    if seams:
        out = fix_seams(out, uvr, original=orig)
    return bleed(out, uvr.mask, padding, keep=orig), info


def flatten_lighting(atlas: ImageLike, uv: UVLayout, groups: Sequence[np.ndarray] | None = None,
                     sigma_frac: float = 0.06, strength: float = 0.8) -> Image.Image:  # fmt: skip
    """Remove large-scale brightness gradients (baked light, vignettes, shadowed halves) from an
    albedo atlas while keeping fine detail. Works per material group (or over all islands) with a
    mask-normalised blur, so neighbouring materials don't bleed into each other's estimate."""
    img = ops.load_image(atlas).convert("RGB")
    uv = uv.resized(img.size)
    arr = np.asarray(img, np.float64)
    lum = arr @ np.array([0.2126, 0.7152, 0.0722])
    sigma = max(6.0, sigma_frac * max(img.size))
    out = arr.copy()
    for g in groups if groups is not None else [uv.mask]:
        if g.sum() < 100:
            continue
        w = g.astype(np.float64)
        low = ndimage.gaussian_filter(lum * w, sigma) / np.maximum(ndimage.gaussian_filter(w, sigma), 1e-6)
        target = lum[g].mean()
        ratio = np.clip(target / np.maximum(low, 1.0), 0.6, 1.6) ** strength
        out[g] = arr[g] * ratio[g][:, None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- #
# Check
# --------------------------------------------------------------------------- #


@dataclass
class Check:
    name: str
    passed: bool
    severity: str  # "error" fails the atlas; "warning" needs a look (usually in the renders)
    value: Any
    detail: str
    islands: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "severity": self.severity, "value": self.value,
                "detail": self.detail, **({"islands": self.islands} if self.islands else {})}  # fmt: skip


def _island_drift(edge_raw: np.ndarray, uv: UVLayout, k: int, search: int) -> dict[str, Any]:
    return _region_drift(edge_raw, uv.islands == k, k, search)


def _region_drift(edge_raw: np.ndarray, region: np.ndarray, k: int, search: int, label: str | None = None) -> dict[str, Any]:
    """Where do the model's edges line up best with a region's true outline (an island, or the part
    of an island given one material)?

    Scores every shift within ±search px by the mean edge strength of the raw output sampled
    at the (shifted) boundary texels. A registered result peaks at (0, 0); ``prominence`` is
    that peak versus the median over all shifts (≈1 means the outline isn't in the output at all).
    """
    edge = np.zeros(region.shape, bool)
    edge[:, :-1] |= region[:, :-1] != region[:, 1:]
    edge[:, 1:] |= region[:, 1:] != region[:, :-1]
    edge[:-1, :] |= region[:-1, :] != region[1:, :]
    edge[1:, :] |= region[1:, :] != region[:-1, :]
    b = edge & region
    ys, xs = np.nonzero(b)
    if len(xs) < 20:
        return {"island": k, "region": label, "skipped": "tiny"}
    if len(xs) > 4000:
        sel = np.random.default_rng(k).choice(len(xs), 4000, replace=False)
        ys, xs = ys[sel], xs[sel]
    H, W = edge_raw.shape
    offs = range(-search, search + 1)
    scores = np.zeros((len(offs), len(offs)))
    for i, dy in enumerate(offs):
        yy = np.clip(ys + dy, 0, H - 1)
        for j, dx in enumerate(offs):
            scores[i, j] = edge_raw[yy, np.clip(xs + dx, 0, W - 1)].mean()
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    zero = scores[search, search]
    med = float(np.median(scores)) + 1e-9
    return {
        "island": k,
        "region": label,
        "best_shift": [int(offs[j]), int(offs[i])],  # (dx, dy) px
        "prominence": round(float(zero / med), 3),
        "peak_prominence": round(float(scores[i, j] / med), 3),
    }


def check(
    new: ImageLike,
    original: ImageLike,
    uv: UVLayout,
    *,
    raw: ImageLike | None = None,
    padding: int = 16,
    max_shift: float | None = None,
    min_prominence: float = 1.25,
    style: ImageLike | None = None,
    backend: Backend | None = None,
    materials: Sequence[tuple[str, str]] | None = None,
    instruction: str = "",
) -> dict[str, Any]:
    """Score a retextured atlas against the rules it must follow to work on the Blender mesh.

    ``raw`` (the model's unclamped output) enables the drift check; without it the check runs on
    ``new``, where clamping hides some drift. ``style`` + ``backend`` add a vision-model review.
    """
    new_img, orig_img = ops.load_image(new), ops.load_image(original)
    checks: list[Check] = []
    W, H = orig_img.size

    # 1. Same pixel dimensions: UVs are normalised, so another size or aspect stretches the texture.
    same = new_img.size == orig_img.size
    checks.append(Check("size_match", same, "error", list(new_img.size),
                        "same WxH as the original atlas" if same else f"original is {W}x{H}; resize back before use"))  # fmt: skip
    if not same:
        new_img = new_img.resize(orig_img.size, Image.Resampling.LANCZOS)
    uv = uv.resized(orig_img.size)
    mask = uv.mask

    # 2. Format: 8-bit RGB(A); alpha only if the original had it.
    alpha_added = new_img.mode in ("RGBA", "LA") and orig_img.mode not in ("RGBA", "LA")
    ok_mode = new_img.mode in ("RGB", "RGBA") and not alpha_added
    pot = all(v & (v - 1) == 0 for v in new_img.size)
    checks.append(Check("format", ok_mode, "warning", {"mode": new_img.mode, "power_of_two": pot},
                        "8-bit RGB(A), no new alpha channel" if ok_mode else f"mode {new_img.mode}"
                        + (" adds an alpha channel the material may treat as transparency" if alpha_added else "")))  # fmt: skip

    n = np.asarray(new_img.convert("RGB"), np.float64)
    o = np.asarray(orig_img.convert("RGB"), np.float64)

    # Drift first (section 5 reports it): coverage uses it to find strips uncovered by offset paint.
    if raw is not None:
        raw_img = ops.load_image(raw).convert("RGB").resize(orig_img.size)
    else:  # un-bled: island texels from the new atlas, the rest from the original
        raw_img = Image.fromarray(np.where(mask[..., None], np.asarray(new_img.convert("RGB")), np.asarray(orig_img.convert("RGB"))), "RGB")
    max_shift = max_shift if max_shift is not None else max(2.0, 0.004 * max(W, H))
    drift = measure_drift(raw_img, uv, materials)
    strips = offset_strips(uv, drift, new_img, orig_img, max_shift, min_prominence, materials)

    # 3. Coverage: every island texel that should be repainted was — none transparent, flat
    #    background, left identical to the original, or inside a strip uncovered by offset paint.
    holes = unpainted(new_img, orig_img, uv, materials) | strips
    hole_frac = float(holes.sum() / max(mask.sum(), 1))
    per_island = {}
    for k in uv.island_ids():
        isl = uv.islands == k
        if isl.sum():
            per_island[k] = float((holes & isl).sum() / isl.sum())
    bad_isl = sorted(k for k, v in per_island.items() if v > 0.005)
    checks.append(Check("coverage", not bad_isl, "error",
                        {"overall": round(1 - hole_frac, 4), "worst_islands": {k: round(v, 4) for k, v in
                         sorted(per_island.items(), key=lambda kv: -kv[1])[:5] if v > 0}},
                        f"{hole_frac:.2%} of island texels unpainted overall; islands over 0.5%: {bad_isl or 'none'} "
                        "(transparent, background colour, or unchanged from the original — often a strip "
                        "left by offset paint; `uv fill-holes` fixes thin strips)", bad_isl))  # fmt: skip

    # 4. Padding bleed: texels just outside islands should carry the nearest island colour, or
    #    mip-mapping/filtering pulls background colour into the seams (dark or bright lines).
    if (~mask).any() and padding > 0:
        dist, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
        ring = (~mask) & (dist <= max(2, padding // 2))
        diff = np.abs(n[ring] - n[iy[ring], ix[ring]]).max(axis=1)
        bled = float((diff < 40).mean()) if ring.any() else 1.0
        checks.append(Check("padding_bleed", bled > 0.9, "error", round(bled, 3),
                            f"{bled:.0%} of the {max(2, padding // 2)}px margin carries island colour (run `imagegen uv bleed`)"))  # fmt: skip

    # 5. Drift: are the model's island outlines where the UV layout says they are? (measured above)
    # Beyond the uncovered strip (counted in coverage), drift moves painted features (eyes, windows,
    # panel lines) off their geometry — an error only where the original has such features.
    e_orig = _edges(orig_img)
    inner_all = ndimage.binary_erosion(mask, iterations=4)
    for d in drift:
        k_inner = inner_all & (uv.islands == d["island"])
        d["has_features"] = bool(k_inner.any() and e_orig[k_inner].mean() > 4)
    drifted = sorted({d["island"] for d in drift if np.hypot(*d["best_shift"]) > max_shift and d["peak_prominence"] > min_prominence})
    lost = sorted({d["island"] for d in drift if d["prominence"] < min_prominence} - set(drifted))
    # Offset paint uncovers a strip on the leading side (coverage catches that directly); on islands
    # whose original has painted features it also moves them off their geometry.
    critical = sorted({d["island"] for d in drift if d["island"] in drifted + lost and d["has_features"]})
    checks.append(Check(
        "layout_registration", not critical, "error",
        {"islands_checked": len(drift), "drifted": drifted, "outline_lost": lost, "with_features": critical,
         "max_shift_px": round(max_shift, 1),
         "source": "raw model output" if raw is not None else "final atlas (pass raw= for a stricter check)"},
        "island content lines up with the UV outlines" if not (drifted or lost) else
        f"drifted {drifted} (model's content offset/rescaled vs the UV island), outline_lost {lost} (model ignored "
        f"the island shape). Error for islands whose original has painted features that must stay registered: "
        f"{critical or 'none'}. On other islands check `coverage` and the renders for an uncovered edge strip", critical,
    ))  # fmt: skip
    if drifted or lost:
        checks.append(Check("layout_spill", False, "warning", sorted(set(drifted + lost) - set(critical)),
                            "model paint was offset/overflowed on featureless islands. Overflow is trimmed by "
                            "clamping, but an offset leaves the leading edge unpainted (see coverage) and cuts "
                            "pattern edges; confirm the island edges on renders",
                            sorted(set(drifted + lost) - set(critical))))  # fmt: skip

    # 6. Features: if the original had detail painted inside islands, it should still be there.
    inner = ndimage.binary_erosion(mask, iterations=4)
    e_o, e_n = _edges(orig_img)[inner], _edges(new_img)[inner]
    if inner.any() and e_o.mean() > 4:
        corr = float(np.corrcoef(e_o, e_n)[0, 1]) if e_n.std() > 0 else 0.0
        checks.append(Check("feature_preservation", corr > 0.15, "warning", round(corr, 3),
                            "correlation of interior detail with the original (low = painted features moved or vanished)"))  # fmt: skip

    # 7. Seams: texels on both sides of a UV seam land next to each other on the mesh. Where the
    #    original was continuous in tone, the new atlas should be too.
    seams = uv.info.get("seams", [])
    if seams:
        sn, so = ndimage.gaussian_filter(n, (2, 2, 0)), ndimage.gaussian_filter(o, (2, 2, 0))
        total = broken = 0.0
        worst: list[tuple[float, dict]] = []
        for s in seams:
            pa, pb = _seam_points(s)
            dn = np.abs(_sample(sn, pa) - _sample(sn, pb)).mean(axis=1)
            do = np.abs(_sample(so, pa) - _sample(so, pb)).mean(axis=1)
            was_cont = do < 15
            bad = was_cont & (dn > 30)
            total += was_cont.sum()
            broken += bad.sum()
            if was_cont.any():
                worst.append((float(dn[was_cont].mean()), s))
        frac = float(broken / total) if total else 0.0
        worst.sort(key=lambda t: -t[0])
        checks.append(Check("seam_continuity", frac < 0.2, "warning", round(frac, 3),
                            f"{frac:.0%} of seam length that was continuous in the original now jumps in tone "
                            f"(run `imagegen uv fix-seams`; check the renders); worst island pairs: "
                            f"{[w[1]['islands'] for w in worst[:5]]}",
                            sorted({k for _, s in worst[:5] for k in s['islands']})))  # fmt: skip

    # 8. Untouched outside: texels far from any island should be byte-identical to the original.
    if (~mask).any():
        far = ndimage.distance_transform_edt(~mask) > padding + 1
        changed = float((np.abs(n[far] - o[far]).max(axis=1) > 2).mean()) if far.any() else 0.0
        checks.append(Check("outside_islands_untouched", changed < 0.01, "warning", round(changed, 4),
                            "fraction of empty-space texels (beyond the padding) that changed"))  # fmt: skip

    # 9. UV layout sanity (from Blender): problems the texture can't fix but that explain artefacts.
    san = uv.info.get("sanity", {})
    issues = []
    if san.get("overlap_texels"):
        issues.append(f"{san['overlap_texels']} texels shared by overlapping islands (mirrored/stacked UVs get identical paint)")
    if san.get("faces_outside_0_1"):
        issues.append(f"{san['faces_outside_0_1']} faces have UVs outside 0-1 (texture repeats there)")
    if san.get("flipped_faces"):
        issues.append(f"{san['flipped_faces']} faces are mirrored in UV space (patterns/text appear flipped)")
    if (san.get("texel_density_cv") or 0) > 0.25:
        issues.append(f"texel density varies (cv={san['texel_density_cv']}): one pattern scale in 2D will look "
                      "bigger/smaller on different parts of the mesh")  # fmt: skip
    if san.get("udim"):
        issues.append("UDIM/tiled image: retexture each tile separately")
    checks.append(Check("uv_layout_sanity", not issues, "warning", san, "; ".join(issues) or "no UV layout problems"))

    # 10. Optional vision review: things only "looking" catches — wrong material on a part,
    #     painted lighting, a picture instead of a texture, text.
    if backend is not None and style is not None:
        imgs = [label_islands(outline(new_img, uv), uv), ops.load_image(style)]
        plan = ""
        orient = uv.orientation()
        if orient is not None:
            imgs.append(orient)
            plan += ("\nImage 3 shows which way each texel's surface faces on the model: blue = up (roofs/tops), "
                     "dark olive = down, red/green/pink/teal = sideways (walls).")  # fmt: skip
        if materials:
            plan += "\nIntended materials by surface: " + "; ".join(f"{sel} -> {d}" for sel, d in materials) + \
                    " (up/down/side refer to the facing directions in image 3; earlier entries win)."  # fmt: skip
        if instruction:
            plan += f"\nArtist direction: {instruction}"
        verdict = backend.judge(_JUDGE_ATLAS + plan, imgs, ATLAS_JUDGE_SCHEMA)
        valid = set(uv.island_ids())
        bad = sorted({int(k) for k in verdict.get("wrong_material_islands", []) if int(k) in valid})
        ok = (verdict["style_match"] >= 6 and not bad and not verdict["picture_not_texture"]
              and not verdict["painted_lighting"] and not verdict["text_or_markings"])  # fmt: skip
        # A warning, not an error: in UV space islands are rotated/mirrored, so a judge looking at the
        # flat atlas misreads correct placements (sideways tile rows on a rotated island). Confirm
        # anything it flags on Blender renders (`imagegen uv render` + `review-renders`) before acting.
        checks.append(Check("vision_review", ok, "warning", verdict,
                            ("; ".join(verdict.get("issues", [])) or "ok")
                            + (f" — flagged islands {bad}: confirm on renders before repairing" if bad else ""), bad))  # fmt: skip

    errors = [c for c in checks if not c.passed and c.severity == "error"]
    warnings = [c for c in checks if not c.passed and c.severity == "warning"]
    repair_islands = sorted({k for c in errors for k in c.islands})
    return {
        "passed": not errors,
        "errors": [c.name for c in errors],
        "warnings": [c.name for c in warnings],
        "repair_islands": repair_islands,
        "checks": [c.to_dict() for c in checks],
        "drift": drift,
    }


ATLAS_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "style_match": {"type": "integer", "description": "0-10 how well materials/colours/detail match image 2"},
        "painted_lighting": {"type": "boolean"},
        "text_or_markings": {"type": "boolean"},
        "picture_not_texture": {"type": "boolean", "description": "drew a scene/object view instead of repainting islands"},
        "pattern_scale_consistent": {"type": "boolean"},
        "wrong_material_islands": {"type": "array", "items": {"type": "integer"},
                                   "description": "island numbers (yellow labels) where a part has the wrong material"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["style_match", "painted_lighting", "text_or_markings", "picture_not_texture", "pattern_scale_consistent",
                 "wrong_material_islands", "issues"],
    "additionalProperties": False,
}

_JUDGE_ATLAS = """\
Image 1 is a UV texture atlas for a 3D model (red lines = UV island outlines and yellow numbers =
island ids; neither is part of the texture). Image 2 is the style reference it should look like.
Judge the atlas as a texture: style_match 0-10 (materials, colours, surface detail vs image 2);
painted_lighting (baked-in shading, shadows, highlights or directional light — bad for an albedo
map); text_or_markings (letters, numbers other than the yellow ids, watermarks); picture_not_texture
(it drew a view of an object or scene instead of repainting the islands); pattern_scale_consistent
(same pattern scale across islands); wrong_material_islands (ids of islands where some part carries
the wrong material for what that part of the model is, e.g. roof tiles on a wall). List concrete issues."""


# --------------------------------------------------------------------------- #
# Render review
# --------------------------------------------------------------------------- #

RENDER_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "style_match": {"type": "integer", "description": "0-10: renders of the retextured model vs the reference"},
        "improved": {"type": "boolean", "description": "after looks closer to the reference than before"},
        "visible_seams": {"type": "boolean"},
        "stretching_or_misalignment": {"type": "boolean"},
        "painted_lighting": {"type": "boolean"},
        "pattern_scale_or_direction_problems": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "fix_suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["style_match", "improved", "visible_seams", "stretching_or_misalignment", "painted_lighting",
                 "pattern_scale_or_direction_problems", "issues", "fix_suggestions"],  # fmt: skip
    "additionalProperties": False,
}


def review_renders(
    after: Sequence[ImageLike],
    style: ImageLike,
    *,
    backend: Backend,
    before: Sequence[ImageLike] = (),
) -> dict[str, Any]:
    """Vision review of Blender renders: the only place seams, stretching and pattern direction
    on the actual mesh become visible."""
    def sheet(paths: Sequence[ImageLike]) -> Image.Image:
        ims = [ops.load_image(p).convert("RGBA") for p in paths]
        cell = ims[0].size
        grid, _ = ops.pack_grid(ims, min(len(ims), 4), cell, padding=4, mode="RGBA")
        bg = Image.new("RGB", grid.size, (128, 128, 128))
        bg.paste(grid, (0, 0), grid)
        return bg

    images = [sheet(after), ops.load_image(style)]
    prompt = ("Image 1 shows renders of a 3D model (several views) with a new texture. Image 2 is the look "
              "the texture should achieve.")  # fmt: skip
    if before:
        images.append(sheet(before))
        prompt += " Image 3 shows the same views with the previous texture."
    prompt += (
        " Judge the texturing on the model: style_match 0-10; improved (vs image 3 if given); visible_seams "
        "(hard lines or tone jumps where UV islands meet); stretching_or_misalignment (features cut off, "
        "smeared, shifted or distorted on the surface); painted_lighting (shading baked into the texture that "
        "fights the scene lighting); pattern_scale_or_direction_problems (e.g. bricks/planks different sizes "
        "on different faces or running the wrong way). List issues and concrete fix suggestions."
    )
    return backend.judge(prompt, images, RENDER_JUDGE_SCHEMA)
