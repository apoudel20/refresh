"""Texture atlases as editable projects.

An atlas project is a directory::

    my-atlas/
      spec.json        what each cell should be (name, prompt, tileable, ...) + atlas settings
      cells/<name>.png one image per cell, at cell_size — the source of truth
      atlas.png        packed grid (rebuilt from cells/ by `repack`)
      atlas.json       frames/UVs per cell (TexturePacker "JSON hash"-compatible)
      history/<name>/  previous versions of a cell, written before every change
      review.json      last `review` report
      log.jsonl        one line per model call / change

Workflow: ``build`` generates missing cells -> ``review`` has a vision model score
each cell against its description (``fix=True`` edits/regenerates the failures) ->
``edit_cell`` / ``revert_cell`` for manual touch-ups -> ``repack``. ``import_atlas``
turns an existing atlas image into a project so it can be edited cell by cell.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from imagegen import ops, pipeline
from imagegen.backends import Backend, GenerationResult
from imagegen.ops import Box, ImageLike

Log = pipeline.Log
_nolog = pipeline._nolog

SEAM_THRESHOLD = 1.6


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass
class CellSpec:
    name: str
    prompt: str = ""
    tileable: bool | None = None  # None -> AtlasSpec.tileable
    reference: str | None = None  # optional image path guiding this cell's content


@dataclass
class AtlasSpec:
    name: str = "atlas"
    cell_size: tuple[int, int] = (256, 256)
    columns: int | None = None  # None -> ceil(sqrt(len(cells)))
    padding: int = 4  # extruded px around each cell in atlas.png (stops filtering bleed)
    style: str = ""  # shared art direction, prepended to every cell prompt
    tileable: bool = False  # default for cells: seamless repeating textures
    background: str = "opaque"  # "opaque" | "transparent" (generated on key_color, then keyed out)
    key_color: str = "#ff00ff"
    mode: str = "cells"  # "cells": one call per cell | "sheet": whole grid in one call
    resample: str = "lanczos"  # "nearest" for pixel art
    power_of_two: bool = False  # pad atlas.png to power-of-two dimensions
    style_reference: str | None = None  # image every cell should match stylistically
    chain_style: bool = True  # no style_reference? use the first finished cell as one
    auto_seamless: bool = True  # blend-fix tileable cells whose seam_score is too high
    cells: list[CellSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: str | Path | None = None) -> "AtlasSpec":
        data = dict(data)
        cells = [CellSpec(**c) if isinstance(c, dict) else CellSpec(name=str(c)) for c in data.pop("cells", [])]
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown atlas spec field(s): {', '.join(sorted(unknown))}")
        spec = cls(**data, cells=cells)
        spec.cell_size = ops.parse_size(spec.cell_size)
        if base_dir is not None:
            spec._resolve_paths(Path(base_dir))
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: str | Path) -> "AtlasSpec":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text()), base_dir=p.parent)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cell_size"] = list(self.cell_size)
        d["cells"] = [{k: v for k, v in c.items() if v is not None} for c in d["cells"]]
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def validate(self) -> None:
        if not self.cells:
            raise ValueError("Atlas spec has no cells")
        seen = set()
        for c in self.cells:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", c.name):
                raise ValueError(f"Cell name {c.name!r} must be letters, digits, '_', '-', '.'")
            if c.name in seen:
                raise ValueError(f"Duplicate cell name {c.name!r}")
            seen.add(c.name)
        if self.mode not in ("cells", "sheet"):
            raise ValueError("mode must be 'cells' or 'sheet'")
        if self.background not in ("opaque", "transparent"):
            raise ValueError("background must be 'opaque' or 'transparent'")

    def _resolve_paths(self, base: Path) -> None:
        def res(p: str | None) -> str | None:
            if not p or p.startswith(("http://", "https://")):
                return p
            q = Path(p).expanduser()
            return str(q if q.is_absolute() else (base / q).resolve())

        self.style_reference = res(self.style_reference)
        for c in self.cells:
            c.reference = res(c.reference)

    # -- derived ------------------------------------------------------------ #

    @property
    def grid(self) -> tuple[int, int]:
        cols = self.columns or math.ceil(math.sqrt(len(self.cells)))
        return cols, math.ceil(len(self.cells) / cols)

    def cell(self, name: str) -> CellSpec:
        for c in self.cells:
            if c.name == name:
                return c
        raise KeyError(f"No cell named {name!r}; cells: {', '.join(c.name for c in self.cells)}")

    def is_tileable(self, cell: CellSpec) -> bool:
        return self.tileable if cell.tileable is None else cell.tileable


EXAMPLE_SPEC: dict[str, Any] = {
    "name": "dungeon-floor",
    "cell_size": 256,
    "columns": 4,
    "padding": 4,
    "style": "hand-painted stylized fantasy game texture, soft top-left lighting, muted palette",
    "tileable": True,
    "background": "opaque",
    "mode": "cells",
    "cells": [
        {"name": "cobblestone", "prompt": "grey cobblestone floor with thin dark mortar lines"},
        {"name": "cobblestone_mossy", "prompt": "the same cobblestone floor with green moss in the cracks"},
        {"name": "dirt", "prompt": "packed brown dirt with small pebbles"},
        {"name": "grass", "prompt": "short lush green grass seen from above"},
        {"name": "wood_planks", "prompt": "worn oak floor planks running horizontally"},
        {"name": "lava", "prompt": "cracked black basalt with glowing orange lava seams"},
        {"name": "water", "prompt": "shallow clear water over sand, gentle ripples"},
        {"name": "bricks", "prompt": "red clay brick wall, running bond pattern"},
    ],
}

EXAMPLE_SPRITES: dict[str, Any] = {
    "name": "items",
    "cell_size": 128,
    "columns": 4,
    "padding": 2,
    "style": "clean 2D game item icons, bold outlines, flat shading",
    "tileable": False,
    "background": "transparent",
    "mode": "cells",
    "cells": [
        {"name": "sword", "prompt": "a short steel sword, diagonal, hilt bottom-left"},
        {"name": "shield", "prompt": "a round wooden shield with an iron rim"},
        {"name": "potion_red", "prompt": "a round glass potion bottle with red liquid and a cork"},
        {"name": "key", "prompt": "an ornate golden key"},
    ],
}


# --------------------------------------------------------------------------- #
# Project directory
# --------------------------------------------------------------------------- #


class AtlasProject:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    @classmethod
    def create(cls, root: str | Path, spec: AtlasSpec) -> "AtlasProject":
        proj = cls(root)
        proj.root.mkdir(parents=True, exist_ok=True)
        (proj.root / "cells").mkdir(exist_ok=True)
        spec.save(proj.spec_path)
        return proj

    @property
    def spec_path(self) -> Path:
        return self.root / "spec.json"

    def spec(self) -> AtlasSpec:
        if not self.spec_path.is_file():
            raise FileNotFoundError(f"{self.root} is not an atlas project (no spec.json)")
        return AtlasSpec.load(self.spec_path)

    def cell_path(self, name: str) -> Path:
        return self.root / "cells" / f"{name}.png"

    def has_cell(self, name: str) -> bool:
        return self.cell_path(name).is_file()

    def load_cell(self, name: str) -> Image.Image:
        return ops.load_image(self.cell_path(name))

    def save_cell(self, name: str, img: Image.Image, reason: str, **info: Any) -> Path:
        """Write a cell, first moving any existing version into history/."""
        path = self.cell_path(name)
        backup = None
        if path.is_file():
            hist = self.root / "history" / name
            hist.mkdir(parents=True, exist_ok=True)
            seq = len(list(hist.glob("*.png"))) + 1  # zero-padded so name order == age order
            backup = hist / f"{seq:04d}-{time.strftime('%Y%m%d-%H%M%S')}.png"
            shutil.copy2(path, backup)
        ops.save_image(img, path)
        self.log(op=reason, cell=name, backup=str(backup) if backup else None, **info)
        return path

    def history(self, name: str) -> list[Path]:
        return sorted((self.root / "history" / name).glob("*.png"))

    def log(self, **entry: Any) -> None:
        entry = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
        with self._lock, open(self.root / "log.jsonl", "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def _gen_info(res: GenerationResult) -> dict[str, Any]:
    return {"backend": res.backend, "model": res.model, "usage": res.usage, **res.meta}


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_TILEABLE_RULES = (
    "Seamless tileable game texture: it must repeat with no visible seams on all four edges. "
    "Flat orthographic top-down view, even lighting (no vignette, no strong cast shadows, no perspective), "
    "fill the entire frame edge to edge, no borders, no text."
)


def _hex(c: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % c


def cell_prompt(spec: AtlasSpec, cell: CellSpec, n_style_refs: int = 0, has_cell_ref: bool = False) -> str:
    lines = []
    if spec.style:
        lines.append(f"Art style: {spec.style}.")
    lines.append(f"Subject: {cell.prompt or cell.name.replace('_', ' ')}.")
    if spec.is_tileable(cell):
        lines.append(_TILEABLE_RULES)
    elif spec.background == "transparent":
        key = _hex(ops.parse_color(spec.key_color))
        lines.append(
            f"One subject, centred and fully inside the frame with a small margin, on a perfectly flat, "
            f"uniform solid {key} background (it will be chroma-keyed out). No shadow, gradient or texture "
            f"on the background, and do not use that colour anywhere in the subject. No text."
        )
    else:
        lines.append("Game art asset that fills the frame. No text, labels, borders or watermarks.")
    idx = 1
    if has_cell_ref:
        lines.append(f"Image {idx} is a content reference for this subject.")
        idx += 1
    if n_style_refs:
        span = f"image {idx}" if n_style_refs == 1 else f"images {idx}-{idx + n_style_refs - 1}"
        lines.append(
            f"Match the art style of {span} exactly (palette, lighting, level of detail, rendering, scale of "
            f"features), but depict the subject above; do not copy that image's content."
        )
    lines.append("Output a square image.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def postprocess_cell(spec: AtlasSpec, cell: CellSpec, img: Image.Image) -> Image.Image:
    """Crop/resize to cell_size, key out the background, and fix seams if needed."""
    # "nearest" keeps pixel art crisp (samples pixel centres instead of averaging).
    out = ops.fit(img.convert("RGB"), spec.cell_size, "cover", spec.resample)
    if spec.is_tileable(cell) and spec.auto_seamless and ops.seam_score(out) > SEAM_THRESHOLD:
        out = ops.seamless_blend(out)
    if spec.background == "transparent" and not spec.is_tileable(cell):
        out = ops.chroma_key(out, ops.parse_color(spec.key_color))
    return out


def _style_refs(spec: AtlasSpec, proj: AtlasProject, exclude: str | None = None) -> list[Image.Image]:
    if spec.style_reference:
        return [ops.load_image(spec.style_reference)]
    if spec.chain_style:
        for c in spec.cells:
            if c.name != exclude and proj.has_cell(c.name):
                return [proj.load_cell(c.name).convert("RGB")]
    return []


def generate_cell(spec: AtlasSpec, proj: AtlasProject, cell: CellSpec, backend: Backend, reason: str = "generate") -> Path:
    style = _style_refs(spec, proj, exclude=cell.name)
    refs = ([ops.load_image(cell.reference)] if cell.reference else []) + style
    prompt = cell_prompt(spec, cell, n_style_refs=len(style), has_cell_ref=bool(cell.reference))
    res = backend.generate(prompt, references=refs, aspect_ratio="1:1" if "1:1" in backend.aspect_ratios else None)
    img = postprocess_cell(spec, cell, res.image)
    return proj.save_cell(cell.name, img, reason, prompt=prompt, **_gen_info(res))


def build(
    spec: AtlasSpec,
    outdir: str | Path,
    backend: Backend,
    *,
    only: Sequence[str] | None = None,
    force: bool = False,
    concurrency: int = 4,
    log: Log = _nolog,
) -> dict[str, Any]:
    """Generate missing cells (or ``only`` these / all with ``force``) and pack the atlas.

    Re-running is cheap: finished cells are kept, so an interrupted build resumes.
    """
    proj = AtlasProject.create(outdir, spec)
    names = set(only) if only else None
    todo = [c for c in spec.cells if (names is None or c.name in names) and (force or not proj.has_cell(c.name))]
    if names and (missing := names - {c.name for c in spec.cells}):
        raise KeyError(f"Unknown cell(s): {', '.join(sorted(missing))}")
    errors: dict[str, str] = {}

    if todo and spec.mode == "sheet":
        _build_sheet(spec, proj, backend, todo, log)
    elif todo:
        # The first cell anchors the style for the rest, so generate it on its own.
        if not _style_refs(spec, proj) and spec.chain_style and len(todo) > 1:
            log(f"[1/{len(todo)}] {todo[0].name} (style anchor)")
            generate_cell(spec, proj, todo[0], backend)
            todo, done = todo[1:], 1
        else:
            done = 0
        total = len(todo) + done

        def work(cell: CellSpec) -> None:
            try:
                generate_cell(spec, proj, cell, backend)
                log(f"  done {cell.name}")
            except Exception as exc:  # keep going; report at the end
                errors[cell.name] = str(exc)
                proj.log(op="error", cell=cell.name, error=str(exc))
                log(f"  FAILED {cell.name}: {exc}")

        if todo:
            log(f"generating {len(todo)} cell(s) of {total} with {backend.name}:{backend.model}, {concurrency} at a time")
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            list(pool.map(work, todo))

    packed = repack(outdir)
    packed["generated"] = sorted({c.name for c in todo} - set(errors)) if spec.mode == "cells" else [c.name for c in todo]
    packed["errors"] = errors
    return packed


_SHEET = """\
Image 1 is a layout guide: a {cols} x {rows} grid of numbered cells on a grey canvas.
Produce a game texture atlas that follows that grid EXACTLY: the same canvas, the same cell
positions and sizes, one asset per cell, each asset filling its own cell and never crossing
into its neighbours. Leave the grey margin outside the grid plain grey.
Do not draw grid lines, numbers, labels or any text in the output.
{style}{tile}
Cells (numbered left-to-right, top-to-bottom):
{cells}"""


def _build_sheet(spec: AtlasSpec, proj: AtlasProject, backend: Backend, todo: list[CellSpec], log: Log) -> None:
    """One call for the whole grid (best style consistency), then slice it into cells."""
    cols, rows = spec.grid
    cw, ch = spec.cell_size
    unit = 1024 / max(cols * cw, rows * ch)
    cell_px = (max(16, round(cw * unit)), max(16, round(ch * unit)))
    gw, gh = cols * cell_px[0], rows * cell_px[1]
    aspect = ops.nearest_aspect(gw, gh, backend.aspect_ratios)
    r = ops.ratio_value(aspect)
    canvas = (max(gw, round(gh * r)), gh) if gw / gh < r else (gw, max(gh, round(gw / r)))
    guide, grid_box = ops.grid_guide(cols, rows, cell_px, [c.name for c in spec.cells], canvas=canvas)

    style = _style_refs(spec, proj) if spec.style_reference else []
    lines = [f"{i + 1}. {c.name}: {c.prompt or c.name}" + (" (seamless tileable)" if spec.is_tileable(c) else "")
             for i, c in enumerate(spec.cells)]  # fmt: skip
    tile_note = (
        "\nCells marked 'seamless tileable' must each wrap seamlessly on their own four edges; "
        "flat top-down view, even lighting." if any(spec.is_tileable(c) for c in spec.cells) else ""
    )
    if spec.background == "transparent":
        tile_note += f"\nNon-tileable assets sit on a flat solid {_hex(ops.parse_color(spec.key_color))} background inside their cell."
    prompt = _SHEET.format(
        cols=cols, rows=rows, style=f"Art style: {spec.style}." if spec.style else "",
        tile=tile_note, cells="\n".join(lines),
    )  # fmt: skip
    if style:
        prompt += "\nMatch the art style of image 2 (palette, lighting, detail), not its content."
    log(f"generating {cols}x{rows} sheet in one call ({aspect})")
    res = backend.generate(prompt, references=[guide, *style], aspect_ratio=aspect)

    sheet_dir = proj.root / "sheet"
    sheet_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ops.save_image(guide, sheet_dir / "guide.png")
    ops.save_image(res.image, sheet_dir / f"raw-{stamp}.png")
    sheet = ops.fit(res.image.convert("RGB"), canvas, "stretch").crop(grid_box)
    inset = max(1, round(min(cell_px) * 0.03))  # trim any stray grid lines the model drew
    tiles = ops.slice_grid(sheet, cols, rows, inset=inset)
    wanted = {c.name for c in todo}
    for cell, tile in zip(spec.cells, tiles):
        if cell.name in wanted:
            proj.save_cell(cell.name, postprocess_cell(spec, cell, tile), "generate-sheet", sheet=f"sheet/raw-{stamp}.png", **_gen_info(res))


# --------------------------------------------------------------------------- #
# Pack
# --------------------------------------------------------------------------- #


def repack(outdir: str | Path) -> dict[str, Any]:
    """Rebuild atlas.png + atlas.json from cells/. Missing cells leave an empty slot."""
    proj = AtlasProject(outdir)
    spec = proj.spec()
    cols, rows = spec.grid
    blank = Image.new("RGBA", spec.cell_size, (0, 0, 0, 0))
    images = [proj.load_cell(c.name) if proj.has_cell(c.name) else blank for c in spec.cells]
    atlas, frames = ops.pack_grid(
        images, cols, spec.cell_size, spec.padding,
        wrap=[spec.is_tileable(c) for c in spec.cells], power_of_two=spec.power_of_two,
    )  # fmt: skip
    ops.save_image(atlas, proj.root / "atlas.png")
    meta = write_atlas_json(
        proj.root / "atlas.json", atlas.size, frames, [c.name for c in spec.cells],
        cell_size=spec.cell_size, padding=spec.padding, columns=cols,
        extra={c.name: {"tileable": spec.is_tileable(c), "prompt": c.prompt} for c in spec.cells},
    )  # fmt: skip
    missing = [c.name for c in spec.cells if not proj.has_cell(c.name)]
    return {"atlas": str(proj.root / "atlas.png"), "json": str(proj.root / "atlas.json"), "size": list(atlas.size),
            "grid": [cols, rows], "cells": len(spec.cells), "missing": missing, **{"meta": meta["meta"]}}  # fmt: skip


def write_atlas_json(
    path: str | Path,
    size: tuple[int, int],
    frames: Sequence[Box],
    names: Sequence[str],
    *,
    cell_size: tuple[int, int],
    padding: int,
    columns: int,
    image_name: str = "atlas.png",
    extra: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Frames in pixels plus normalised UVs (origin top-left; ``uv_gl`` has v flipped for OpenGL)."""
    W, H = size
    out: dict[str, Any] = {"frames": {}, "meta": {}}
    for i, (name, (x, y, w, h)) in enumerate(zip(names, frames)):
        out["frames"][name] = {
            "frame": {"x": x, "y": y, "w": w, "h": h},
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": w, "h": h},
            "sourceSize": {"w": w, "h": h},
            "index": i,
            "col": i % columns,
            "row": i // columns,
            "uv": [x / W, y / H, (x + w) / W, (y + h) / H],
            "uv_gl": [x / W, 1 - (y + h) / H, (x + w) / W, 1 - y / H],
            **((extra or {}).get(name, {})),
        }
    out["meta"] = {
        "app": "imagegen",
        "image": image_name,
        "format": "RGBA8888",
        "size": {"w": W, "h": H},
        "cell_size": {"w": cell_size[0], "h": cell_size[1]},
        "padding": padding,
        "columns": columns,
        "rows": math.ceil(len(frames) / columns),
    }
    Path(path).write_text(json.dumps(out, indent=2) + "\n")
    return out


def pack_images(
    images: Sequence[str | Path],
    out: str | Path,
    *,
    columns: int | None = None,
    cell_size: tuple[int, int] | None = None,
    padding: int = 0,
    tileable: bool = False,
    power_of_two: bool = False,
) -> dict[str, Any]:
    """Pack loose image files into ``out`` (+ ``out`` with .json). Names come from file stems."""
    paths = [Path(p) for p in images]
    ims = [ops.load_image(p) for p in paths]
    cell = cell_size or ims[0].size
    cols = columns or math.ceil(math.sqrt(len(ims)))
    atlas, frames = ops.pack_grid(ims, cols, cell, padding, wrap=[tileable] * len(ims), power_of_two=power_of_two)
    out = Path(out)
    ops.save_image(atlas, out)
    write_atlas_json(out.with_suffix(".json"), atlas.size, frames, [p.stem for p in paths],
                     cell_size=cell, padding=padding, columns=cols, image_name=out.name)  # fmt: skip
    return {"atlas": str(out), "json": str(out.with_suffix(".json")), "size": list(atlas.size), "cells": len(ims)}


def slice_image(
    image: ImageLike,
    grid: tuple[int, int],
    outdir: str | Path,
    *,
    names: Sequence[str] | None = None,
    source_padding: int = 0,
) -> list[Path]:
    """Cut an atlas/sprite sheet into cell files (``source_padding`` strips per-slot padding)."""
    cols, rows = grid
    tiles = ops.slice_grid(ops.load_image(image), cols, rows, inset=source_padding)
    names = list(names or [f"r{i // cols}c{i % cols}" for i in range(len(tiles))])
    if len(names) < len(tiles):
        names += [f"r{i // cols}c{i % cols}" for i in range(len(names), len(tiles))]
    outdir = Path(outdir)
    return [ops.save_image(t, outdir / f"{n}.png") for n, t in zip(names, tiles)]


def import_atlas(
    image: ImageLike,
    grid: tuple[int, int],
    outdir: str | Path,
    *,
    names: Sequence[str] | None = None,
    prompts: Sequence[str] | None = None,
    source_padding: int = 0,
    padding: int = 0,
    style: str = "",
    tileable: bool = False,
    background: str = "opaque",
) -> dict[str, Any]:
    """Turn an existing atlas image into an editable project (then edit_cell / review / repack)."""
    cols, rows = grid
    img = ops.load_image(image)
    tiles = ops.slice_grid(img, cols, rows, inset=source_padding)
    names = list(names or [f"r{i // cols}c{i % cols}" for i in range(len(tiles))])[: len(tiles)]
    prompts = list(prompts or [])
    spec = AtlasSpec(
        name=Path(str(image)).stem if isinstance(image, (str, Path)) else "imported",
        cell_size=tiles[0].size, columns=cols, padding=padding, style=style, tileable=tileable,
        background=background, cells=[CellSpec(name=n, prompt=prompts[i] if i < len(prompts) else "") for i, n in enumerate(names)],
    )  # fmt: skip
    spec.validate()
    proj = AtlasProject.create(outdir, spec)
    for n, t in zip(names, tiles):
        proj.save_cell(n, t, "import", source=str(image) if isinstance(image, (str, Path)) else None)
    return repack(outdir)


# --------------------------------------------------------------------------- #
# Edit / revert / upscale
# --------------------------------------------------------------------------- #


def edit_cell(
    outdir: str | Path,
    name: str,
    backend: Backend | None = None,
    *,
    instruction: str | None = None,
    prompt: str | None = None,
    image: ImageLike | None = None,
    region: Box | None = None,
    mask: ImageLike | None = None,
    references: Sequence[ImageLike] = (),
    log: Log = _nolog,
) -> dict[str, Any]:
    """Change one cell, then repack. Exactly one of:

    * ``instruction`` — edit the current cell image (optionally only ``region``/``mask``, in cell pixels)
    * ``prompt``      — replace the cell's description in spec.json and regenerate it
    * ``image``       — drop in your own image
    """
    if sum(x is not None for x in (instruction, prompt, image)) != 1:
        raise ValueError("Pass exactly one of instruction, prompt, image")
    proj = AtlasProject(outdir)
    spec = proj.spec()
    cell = spec.cell(name)

    if image is not None:
        proj.save_cell(name, postprocess_cell(spec, cell, ops.load_image(image)), "replace",
                       source=str(image) if isinstance(image, (str, Path)) else None)  # fmt: skip
    elif prompt is not None:
        if backend is None:
            raise ValueError("Regenerating needs a backend")
        cell.prompt = prompt
        spec.save(proj.spec_path)
        log(f"regenerating {name}")
        generate_cell(spec, proj, cell, backend, reason="regenerate")
    else:
        if backend is None:
            raise ValueError("Editing needs a backend")
        current = proj.load_cell(name)
        work = current.convert("RGB")
        if spec.background == "transparent" and current.mode == "RGBA" and not spec.is_tileable(cell):
            # Give the model the keyed-out area back as the key colour, so it can keep it flat.
            key = Image.new("RGB", current.size, ops.parse_color(spec.key_color))
            work = Image.composite(current.convert("RGB"), key, current.getchannel("A"))
        extra = ""
        if spec.is_tileable(cell):
            extra = " The texture must stay seamless and tileable."
        elif spec.background == "transparent":
            extra = f" Keep the flat {_hex(ops.parse_color(spec.key_color))} background untouched."
        res = pipeline.edit(work, instruction + extra, backend=backend, region=region, mask=mask,
                            references=references, log=log)  # fmt: skip
        out = res.image
        if region is None and mask is None:
            out = postprocess_cell(spec, cell, out)
        else:
            if spec.is_tileable(cell) and spec.auto_seamless and ops.seam_score(out) > SEAM_THRESHOLD:
                out = ops.seamless_blend(out)
            if spec.background == "transparent" and not spec.is_tileable(cell):
                out = ops.chroma_key(out, ops.parse_color(spec.key_color))
        info = res.info()
        proj.save_cell(name, out, "edit", instruction=instruction, region=info["region"], calls=info["calls"])
    packed = repack(outdir)
    return {"cell": name, "path": str(proj.cell_path(name)), "history": len(proj.history(name)), "atlas": packed["atlas"]}


def revert_cell(outdir: str | Path, name: str, steps: int = 1) -> dict[str, Any]:
    """Restore a previous version from history/ (the current one is kept in history too)."""
    proj = AtlasProject(outdir)
    proj.spec().cell(name)
    hist = proj.history(name)
    if len(hist) < steps:
        raise ValueError(f"{name} has only {len(hist)} version(s) in history")
    target = ops.load_image(hist[-steps])
    proj.save_cell(name, target, "revert", restored=str(hist[-steps]))
    repack(outdir)
    return {"cell": name, "restored": str(hist[-steps])}


def upscale_atlas(
    outdir: str | Path,
    scale: float,
    *,
    method: str = "lanczos",
    backend: Backend | None = None,
    concurrency: int = 4,
    log: Log = _nolog,
) -> dict[str, Any]:
    """Upscale every cell independently (so the grid stays exact) and update cell_size."""
    proj = AtlasProject(outdir)
    spec = proj.spec()
    new_size = (round(spec.cell_size[0] * scale), round(spec.cell_size[1] * scale))

    def work(cell: CellSpec) -> None:
        if not proj.has_cell(cell.name):
            return
        big = pipeline.upscale(proj.load_cell(cell.name), scale, method=method, backend=backend,
                               hint=cell.prompt, concurrency=1, log=log)  # fmt: skip
        big = ops.fit(big, new_size, "stretch")
        if spec.is_tileable(cell) and ops.seam_score(big) > SEAM_THRESHOLD:
            big = ops.seamless_blend(big, band=0.1)
        proj.save_cell(cell.name, big, "upscale", scale=scale, method=method)
        log(f"  upscaled {cell.name}")

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(work, spec.cells))
    spec.cell_size = new_size
    spec.save(proj.spec_path)
    return repack(outdir)


# --------------------------------------------------------------------------- #
# Review
# --------------------------------------------------------------------------- #

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "0-10: how well the cell meets its description and rules"},
        "matches": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "fix_instruction": {"type": "string", "description": "one concrete edit instruction, or empty"},
    },
    "required": ["score", "matches", "issues", "fix_instruction"],
    "additionalProperties": False,
}

CONSISTENCY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "outliers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["name", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["outliers"],
    "additionalProperties": False,
}

_REVIEW = """\
You are reviewing one cell of a game texture atlas.
Cell name: {name}
It should depict: {prompt}
{style}{rules}
Score 0-10 how well image 1 meets the description AND the rules (10 = ship it). List concrete
issues. If the score is below 10, give ONE short, specific edit instruction that an image-editing
model could apply to image 1 to fix the most important problem; otherwise an empty string."""


def _checker(img: Image.Image, square: int = 16) -> Image.Image:
    """Show transparency as a grey checkerboard (what a reviewer model can actually see)."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    yy, xx = np.mgrid[0 : img.height, 0 : img.width]
    board = np.where(((xx // square + yy // square) % 2)[..., None] == 1, 240, 200).astype(np.uint8)
    bg = Image.fromarray(np.repeat(board, 3, axis=2), "RGB")
    bg.paste(img, (0, 0), img)
    return bg


def review_cell(spec: AtlasSpec, proj: AtlasProject, cell: CellSpec, backend: Backend) -> dict[str, Any]:
    img = proj.load_cell(cell.name)
    view = _checker(img)
    images = [view]
    rules = []
    if spec.is_tileable(cell):
        images.append(ops.tile_preview(view, 2))
        rules.append("It must tile seamlessly: image 2 is image 1 repeated 2x2 — look for visible seams, "
                     "lines or obvious repetition at the joins.")  # fmt: skip
    elif spec.background == "transparent":
        rules.append("Transparent areas are shown as a grey checkerboard; the subject should be complete, "
                     "centred, cleanly cut out, with no leftover background colour.")  # fmt: skip
    rules.append("No text, watermark, border or frame. Readable at small size.")
    verdict = backend.judge(
        _REVIEW.format(
            name=cell.name, prompt=cell.prompt or cell.name.replace("_", " "),
            style=f"Art style: {spec.style}\n" if spec.style else "", rules="Rules: " + " ".join(rules),
        ),  # fmt: skip
        images,
        REVIEW_SCHEMA,
    )
    verdict["seam_score"] = round(ops.seam_score(img), 3) if spec.is_tileable(cell) else None
    return verdict


def review(
    outdir: str | Path,
    backend: Backend,
    *,
    fix: bool = False,
    threshold: int = 7,
    rounds: int = 1,
    consistency: bool = True,
    only: Sequence[str] | None = None,
    concurrency: int = 4,
    log: Log = _nolog,
) -> dict[str, Any]:
    """Have a vision model score each cell; with ``fix``, repair failures and re-check.

    Low scores (<= 3) are regenerated from the prompt; others get the reviewer's edit
    instruction applied. Tileable cells with a high ``seam_score`` get a seam blend.
    ``consistency`` adds one call over the whole atlas to flag style outliers.
    """
    proj = AtlasProject(outdir)
    spec = proj.spec()
    cells = [c for c in spec.cells if proj.has_cell(c.name) and (not only or c.name in only)]
    report: dict[str, Any] = {"threshold": threshold, "rounds": [], "cells": {}}

    for rnd in range(max(1, rounds) if fix else 1):
        log(f"review round {rnd + 1}: {len(cells)} cell(s)")

        def check(cell: CellSpec) -> tuple[str, dict[str, Any]]:
            try:
                return cell.name, review_cell(spec, proj, cell, backend)
            except Exception as exc:
                return cell.name, {"error": str(exc)}

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            results = dict(pool.map(check, cells))
        report["cells"].update(results)

        outliers: list[dict[str, str]] = []
        if consistency and rnd == 0 and len(spec.cells) > 1:
            outliers = _consistency(spec, proj, backend)
            report["style_outliers"] = outliers

        failing = [
            c for c in cells
            if "error" not in results[c.name]
            and (results[c.name]["score"] < threshold
                 or (results[c.name].get("seam_score") or 0) > SEAM_THRESHOLD)
        ]  # fmt: skip
        outlier_names = {o["name"] for o in outliers} - {c.name for c in failing}
        report["rounds"].append({
            "round": rnd + 1,
            "scores": {n: r.get("score") for n, r in results.items()},
            "failing": [c.name for c in failing],
            "style_outliers": sorted(outlier_names),
        })  # fmt: skip
        log("  scores: " + ", ".join(f"{n}={r.get('score', 'ERR')}" for n, r in results.items()))
        if not fix or (not failing and not outlier_names):
            break

        def repair(cell: CellSpec) -> None:
            r = results[cell.name]
            try:
                if r["score"] < threshold and r["score"] <= 3:
                    log(f"  regenerating {cell.name} (score {r['score']})")
                    generate_cell(spec, proj, cell, backend, reason="review-regenerate")
                elif r["score"] < threshold and r.get("fix_instruction"):
                    log(f"  editing {cell.name}: {r['fix_instruction']}")
                    edit_cell(outdir, cell.name, backend, instruction=r["fix_instruction"])
                if spec.is_tileable(cell):
                    img = proj.load_cell(cell.name)
                    if ops.seam_score(img) > SEAM_THRESHOLD:
                        proj.save_cell(cell.name, ops.seamless_blend(img), "review-seam")
            except Exception as exc:
                proj.log(op="error", cell=cell.name, error=str(exc))
                log(f"  FAILED to fix {cell.name}: {exc}")

        def restyle(name: str) -> None:
            anchor = next((c.name for c in spec.cells if c.name != name and c.name not in outlier_names and proj.has_cell(c.name)), None)
            if not anchor:
                return
            reason = next(o["reason"] for o in outliers if o["name"] == name)
            log(f"  restyling {name}: {reason}")
            try:
                edit_cell(outdir, name, backend, references=[proj.load_cell(anchor).convert("RGB")],
                          instruction=f"Restyle this to match the art style of image 2 (palette, lighting, "
                                      f"detail level, rendering) while keeping the same subject. Problem: {reason}")  # fmt: skip
            except Exception as exc:
                log(f"  FAILED to restyle {name}: {exc}")

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            list(pool.map(repair, failing))
            list(pool.map(restyle, sorted(outlier_names)))
        cells = failing + [spec.cell(n) for n in sorted(outlier_names)]

    repack(outdir)
    (proj.root / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    proj.log(op="review", fix=fix, rounds=len(report["rounds"]))
    report["path"] = str(proj.root / "review.json")
    return report


def _consistency(spec: AtlasSpec, proj: AtlasProject, backend: Backend) -> list[dict[str, str]]:
    names = [c.name for c in spec.cells if proj.has_cell(c.name)]
    ims = [_checker(proj.load_cell(n)) for n in names]
    cols = math.ceil(math.sqrt(len(ims)))
    sheet, _ = ops.pack_grid(ims, cols, spec.cell_size, padding=max(2, spec.cell_size[0] // 32), mode="RGB")
    legend = ", ".join(f"{i + 1}={n}" for i, n in enumerate(names))
    try:
        res = backend.judge(
            f"Image 1 is a texture atlas, cells numbered left-to-right, top-to-bottom: {legend}. "
            f"Intended shared art style: {spec.style or 'consistent across all cells'}. "
            "Which cells clearly do not match the others in art style (palette, lighting direction, "
            "detail level, rendering technique, feature scale)? Return only clear outliers, by cell "
            "name, with a short reason. Return an empty list if the set is consistent.",
            [sheet],
            CONSISTENCY_SCHEMA,
        )
    except Exception as exc:
        proj.log(op="error", step="consistency", error=str(exc))
        return []
    valid = set(names)
    return [o for o in res.get("outliers", []) if o.get("name") in valid]
