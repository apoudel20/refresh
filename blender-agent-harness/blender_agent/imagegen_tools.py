"""imagegen-skill tools for the agent: create images, edit images, retexture a UV atlas.

The backend is chosen by imagegen (``IMAGEGEN_BACKEND``; default: Codex subscription when
installed, else OpenRouter, or ``openai``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def _backend(name: str | None = None, model: str | None = None):
    from imagegen.backends import get_backend

    return get_backend(name, model)


def generate_image(prompt: str, output_path: str, references: list[str] | None = None, size: int | None = None,
                   backend: str | None = None) -> dict[str, Any]:
    from imagegen import ops, pipeline

    images, _ = pipeline.generate(prompt, backend=_backend(backend), references=references or (),
                                  size=(size, size) if size else None)
    path = ops.save_image(images[0], output_path)
    return {"path": str(path), "_images": [str(path)]}


def edit_image(image_path: str, instruction: str, output_path: str, region: list[float] | None = None,
               backend: str | None = None) -> dict[str, Any]:
    from imagegen import ops, pipeline

    box = None
    if region:
        w, h = Image.open(image_path).size
        x0, y0, x1, y1 = region
        if max(region) <= 1.0:
            x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
        box = (int(x0), int(y0), int(x1), int(y1))
    result = pipeline.edit(image_path, instruction, backend=_backend(backend), region=box)
    path = ops.save_image(result.image, output_path)
    return {"path": str(path), "_images": [str(path)]}


def retexture_uv(connector: Any, object_name: str, style_image_path: str, work_dir: str | Path,
                 materials: list[str] | None = None, instruction: str = "", backend: str | None = None) -> dict[str, Any]:
    """Snapshot the live object, export its UV layout in background Blender, repaint the atlas with
    imagegen's constrained UV pipeline, then assign the new atlas to the live object."""
    from imagegen import blender_bridge, ops, uvtex

    work = Path(work_dir) / f"retexture_{object_name}"
    work.mkdir(parents=True, exist_ok=True)
    snap = work / "object.blend"
    connector.snapshot(str(snap))
    uv_dir = work / "uv"
    blender_bridge.export_uv(snap, uv_dir, obj=object_name)
    uv = uvtex.UVLayout.load(uv_dir)
    atlas_path = next((t["filepath"] for t in uv.info.get("textures", []) if t.get("filepath")), None)
    if not atlas_path or not Path(atlas_path).is_file():
        atlas_path = str(work / "placeholder.png")
        uvtex.placeholder(uv).save(atlas_path)
    b = _backend(backend)
    mats = uvtex.parse_materials(materials)
    res = uvtex.retexture(atlas_path, uv, [style_image_path], backend=b, instruction=instruction,
                          materials=mats, padding=16, mode="auto")
    final, _ = uvtex.finish(res.image, atlas_path, uv, materials=mats, padding=16, raw=res.raw)
    out = ops.save_image(final, work / "atlas.png")
    connector.set_material(object_name, str(out), "UV")
    return {"atlas": str(out), "groups": res.groups, "_images": [str(out)]}
