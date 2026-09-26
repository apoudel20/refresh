"""
POST /eval — run the evaluator against a set of render images.

Usage:
    uvicorn eval_api:app --reload

Request (multipart/form-data):
    renders[]       one or more render PNG files  (required)
    depth           depth map PNG                 (optional)
    ply             PLY mesh file                 (optional)
    reference_image reference image for comparison (optional)
    backend         "render_eval" | "openai" | "openrouter" | "claude" | "codex"  (default: render_eval)
    model           model override                (optional)

Response:
    {
        "overall_score": 0.82,
        "visual_fidelity": 0.85,
        "topology_quality": 0.78,
        "depth_alignment": 0.80,
        "vertex_accuracy": 0.75,
        "feedback": ["...", ...]
    }
"""

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

import sys
sys.path.insert(0, str(Path(__file__).parent))

from blender_agent.evaluator import EvaluatorClient, RenderPayload

app = FastAPI(title="Render Evaluator")


def _build_client(backend: str, model: str | None) -> EvaluatorClient:
    if backend == "render_eval":
        return EvaluatorClient.from_render_eval(fast=False)
    if backend == "claude":
        return EvaluatorClient.from_claude(**({"model": model} if model else {}))
    if backend == "openrouter":
        return EvaluatorClient.from_openrouter(**({"model": model} if model else {}))
    if backend == "codex":
        return EvaluatorClient.from_codex(model=model)
    return EvaluatorClient.from_openai(**({"model": model} if model else {}))


@app.post("/eval")
async def eval_renders(
    renders: Annotated[list[UploadFile], File()],
    depth: Annotated[UploadFile | None, File()] = None,
    ply: Annotated[UploadFile | None, File()] = None,
    reference_image: Annotated[UploadFile | None, File()] = None,
    backend: Annotated[str, Form()] = "render_eval",
    model: Annotated[str | None, Form()] = None,
):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        render_paths = []
        for i, f in enumerate(renders):
            p = tmp / f"render_{i}.png"
            p.write_bytes(await f.read())
            render_paths.append(str(p))

        depth_path = None
        if depth:
            depth_path = str(tmp / "depth.png")
            (tmp / "depth.png").write_bytes(await depth.read())

        ply_path = None
        if ply:
            ply_path = str(tmp / "mesh.ply")
            (tmp / "mesh.ply").write_bytes(await ply.read())

        reference = {}
        if reference_image:
            ref_path = str(tmp / "reference.png")
            (tmp / "reference.png").write_bytes(await reference_image.read())
            reference["image_path"] = ref_path

        payload = RenderPayload(
            render_images=render_paths,
            depth_map_path=depth_path,
            ply_path=ply_path,
        )

        client = _build_client(backend, model)
        result = client.evaluate(payload, reference)

    return JSONResponse({
        "overall_score":    result.overall_score,
        "visual_fidelity":  result.visual_fidelity,
        "topology_quality": result.topology_quality,
        "depth_alignment":  result.depth_alignment,
        "vertex_accuracy":  result.vertex_accuracy,
        "feedback":         result.feedback,
    })
