"""The harness-owned stage: a locked camera and lights every scored render uses."""

from __future__ import annotations

from typing import Any

from lineage.hashing import H, content_hash
from PIL import Image


def stage_spec(reference_path: str, long_side: int = 768) -> dict[str, Any]:
    """Front-on camera framing a ~2 m object at the origin, 3-point lights, reference aspect ratio."""
    w, h = Image.open(reference_path).size
    scale = long_side / max(w, h)
    return {
        "resolution": [max(64, round(w * scale)), max(64, round(h * scale))],
        "camera": {"location": [0.0, -6.0, 1.0], "target": [0.0, 0.0, 1.0], "lens": 50.0},
        "lights": [
            {"type": "AREA", "location": [4.0, -5.0, 5.0], "energy": 800.0, "size": 4.0},
            {"type": "AREA", "location": [-5.0, -3.0, 2.0], "energy": 300.0, "size": 4.0},
            {"type": "AREA", "location": [0.0, 5.0, 4.0], "energy": 400.0, "size": 3.0},
        ],
        "engine": "EEVEE",
        "world_strength": 0.6,
    }


def task_hash(reference_path: str, spec: dict[str, Any]) -> str:
    """Identity of the task: the reference pixels plus the stage (a different camera is a different task)."""
    with open(reference_path, "rb") as f:
        return H("task", content_hash(f.read()), spec)
