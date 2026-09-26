"""
The reference photo is for looking at, not for pasting.

Every score compares the stage render with the reference photo, so a model that projects the photo, or a relief
extruded from its pixels, scores high without being a model of anything. The guard keeps the photo out of
Blender: tool calls that would bring a reference view (or an image edited or generated from one) into the scene
are refused, and a scene that still holds one is disqualified when it is scored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

RULE = ("The reference photo is for looking at only. Never load it, or an image edited or generated from it, into "
        "Blender; never project it or use it as a texture; never build geometry from its pixels (silhouette "
        "extrusion, relief, displacement, depth maps). Capture the object's essence and model it in 3-D, all the "
        "way round. Scenes that contain the photo score 0.")

_digests: dict[tuple[str, int, int], str] = {}


class ReferenceGuard:
    def __init__(self, reference: dict[str, Any]):
        views = [reference.get("image_path"), *(reference.get("extra_views") or [])]
        self.views = [Path(p).resolve() for p in views if p]
        self.folder = self.views[0].parent if self.views else None
        self.ledger = self.folder / "derived_images.json" if self.folder else None

    def protected(self) -> list[Path]:
        derived: list[Path] = []
        if self.ledger and self.ledger.is_file():
            try:
                derived = [Path(p) for p in json.loads(self.ledger.read_text())]
            except (OSError, ValueError):
                pass
        return self.views + derived

    def is_protected(self, path: str | Path | None) -> bool:
        if not path or not self.views:
            return False
        try:
            p = Path(path).expanduser().resolve()
        except (OSError, RuntimeError):
            return False
        prot = self.protected()
        if p in prot or p.parent == self.folder:
            return True
        d = _digest(p)
        return d is not None and d in {_digest(q) for q in prot}

    def mark_derived(self, path: str | Path | None) -> None:
        if not path or not self.ledger:
            return
        items = [str(p) for p in self.protected()[len(self.views):]]
        p = str(Path(path).resolve())
        if p not in items:
            self.ledger.write_text(json.dumps(items + [p], indent=1))

    def check_path(self, path: str | Path | None) -> None:
        if self.is_protected(path):
            raise PermissionError(f"{path} is the reference photo or an image made from it. {RULE}")

    def check_code(self, code: str) -> None:
        """Refuse Blender code that names a reference image or the folder holding the reference views."""
        names = {str(p) for p in self.protected()} | {p.name for p in self.protected()}
        if self.folder:
            names.add(str(self.folder))
        hit = next((n for n in sorted(names, key=len, reverse=True) if n and n in code), None)
        if hit:
            raise PermissionError(f"Blender code may not read {hit}. {RULE}")

    def scene_violations(self, images: Iterable[dict[str, Any]]) -> list[str]:
        """Names of Blender images that load a reference view or an image made from one."""
        return [img.get("name", "?") for img in images if self.is_protected(img.get("path"))]


def _digest(path: Path) -> str | None:
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        if key not in _digests:
            _digests[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return _digests[key]
    except OSError:
        return None
