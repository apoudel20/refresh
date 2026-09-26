"""
Is it a 3-D object, or a picture of one?

render-eval scores the model from the stage camera only, so a relief or an open shell shaped to that one view
(half a dog, with nothing behind it) can match the photo almost perfectly. This check looks from all around:
``BlenderMCPConnector.turntable`` renders the model from 8 sides with front faces slate blue and back faces red.

  closure    no red from any side: the surface is closed, there is a back to the object
  thickness  from the sides the model is not a sliver: it has real depth, not a relief
  solidity   0.6 * closure + 0.4 * thickness

``fitness_factor`` turns solidity into the multiplier the harness applies to the render-eval composite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

THICK_TARGET = 0.35   # side silhouette area / front silhouette area that counts as fully 3-D
FLOOR = 0.25          # fitness multiplier for a model with no depth or no back at all


def measure(connector: Any, out_dir: str | Path, views: int = 8, size: int = 256) -> dict[str, Any]:
    out_dir = Path(out_dir)
    res = connector.turntable(str(out_dir), views=views, size=size) or {}
    if res.get("empty") or not res.get("views"):
        return {"solidity": 0.0, "closure": 0.0, "thickness": 0.0, "views": [], "sheet": None,
                "feedback": ["The scene has no model yet."]}
    per_view = []
    for v in res["views"]:
        rgba = np.asarray(Image.open(v["path"]).convert("RGBA"), dtype=np.float32) / 255.0
        cover = rgba[..., 3] > 0.5
        back = cover & (rgba[..., 0] > 0.5) & (rgba[..., 1] < 0.35)
        area = int(cover.sum())
        per_view.append({"azimuth": v["azimuth"], "path": v["path"], "area": area,
                         "backface": float(back.sum() / area) if area else 0.0})
    n = len(per_view)
    fronts = [per_view[0]["area"], per_view[n // 2]["area"]]
    sides = [per_view[n // 4]["area"], per_view[3 * n // 4]["area"]] if n >= 4 else [per_view[-1]["area"]]
    thickness = float(min(1.0, (min(sides) / max(max(fronts), 1)) / THICK_TARGET))
    bf = [v["backface"] for v in per_view]
    closure = float(max(0.0, 1.0 - (0.5 * float(np.mean(bf)) + 0.5 * max(bf))))
    solidity = 0.6 * closure + 0.4 * thickness
    worst = max(per_view, key=lambda v: v["backface"])
    feedback = []
    if worst["backface"] > 0.05:
        feedback.append(f"solidity: from azimuth {worst['azimuth']:.0f} deg, {worst['backface']:.0%} of what you see "
                        "is the inside of the surface (red in the turntable): the model is an open shell. Model the "
                        "far side too and close the surface.")
    if thickness < 0.8:
        feedback.append(f"solidity: from the side the model is {min(sides) / max(max(fronts), 1):.0%} as wide as "
                        "from the front; it is too flat. Give it real depth, as the object would have.")
    return {"solidity": round(solidity, 4), "closure": round(closure, 4), "thickness": round(thickness, 4),
            "views": per_view, "sheet": str(_sheet(per_view, out_dir / "turntable.png")), "feedback": feedback}


def fitness_factor(solidity: float) -> float:
    return FLOOR + (1.0 - FLOOR) * max(0.0, min(1.0, solidity))


def _sheet(per_view: list[dict[str, Any]], path: Path) -> Path:
    """All views side by side on a light background (back faces stay red)."""
    tiles = [Image.open(v["path"]).convert("RGBA") for v in per_view]
    w, h = tiles[0].size
    sheet = Image.new("RGBA", (w * len(tiles), h), (236, 239, 244, 255))
    for i, t in enumerate(tiles):
        shaded = Image.new("RGBA", t.size, (236, 239, 244, 255))
        shaded.alpha_composite(t)
        sheet.paste(shaded, (i * w, 0))
    sheet.convert("RGB").save(path)
    return path
