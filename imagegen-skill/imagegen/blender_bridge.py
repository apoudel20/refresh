"""Run the scripts in imagegen/blender/ with a Blender binary (background mode, never saves the .blend)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from imagegen.backends import ImageGenError

SCRIPTS = Path(__file__).resolve().parent / "blender"
_MAC_APP = "/Applications/Blender.app/Contents/MacOS/Blender"


def find_blender() -> str:
    for cand in (os.getenv("BLENDER"), shutil.which("blender"), _MAC_APP):
        if cand and Path(cand).exists():
            return cand
    raise ImageGenError("Blender not found. Install it, put `blender` on PATH, or set BLENDER=/path/to/blender.")


def run_script(script: str, blend: str | Path | None, args: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
    """Run ``imagegen/blender/<script>`` and return the JSON from its ``*_OK`` line."""
    cmd = [find_blender(), "-b"]
    if blend:
        if not Path(blend).is_file():
            raise FileNotFoundError(f".blend not found: {blend}")
        cmd.append(str(blend))
    cmd += ["--factory-startup", "--python-exit-code", "1", "--python", str(SCRIPTS / script), "--"]
    for key, val in args.items():
        if val is not None:
            cmd += [f"--{key.replace('_', '-')}", str(val)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ImageGenError(f"Blender timed out after {timeout:.0f}s running {script}") from exc
    for line in reversed(proc.stdout.splitlines()):
        if "_OK " in line:
            return json.loads(line.split("_OK ", 1)[1])
    tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-25:])
    raise ImageGenError(f"Blender {script} failed (exit {proc.returncode}):\n{tail}", detail=tail)


def export_uv(blend: str | Path, out_dir: str | Path, *, obj: str | None = None, uv_map: str | None = None,
              size: str | None = None) -> dict[str, Any]:  # fmt: skip
    return run_script("uv_export.py", blend, {"out_dir": out_dir, "object": obj, "uv_map": uv_map, "size": size})


def render(blend: str | Path, out_dir: str | Path, *, image: str | Path | None = None, obj: str | None = None,
           replace: str | None = None, views: str = "front,right,back,iso", res: int = 768,
           engine: str = "eevee", samples: int = 32, save: str | None = None, opaque: bool = False) -> dict[str, Any]:  # fmt: skip
    args = {"out_dir": out_dir, "image": Path(image).resolve() if image else None, "object": obj, "replace": replace,
            "views": views, "res": res, "engine": engine, "samples": samples, "save": save,
            "transparent": "0" if opaque else "1"}  # fmt: skip
    return run_script("apply_render.py", blend, args)


def make_demo_scene(out_dir: str | Path, size: int = 1024) -> dict[str, Any]:
    return run_script("make_demo_scene.py", None, {"out_dir": out_dir, "size": size})
