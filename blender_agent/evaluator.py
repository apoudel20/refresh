"""
Evaluator API interface.

BlenderAgent calls EvaluatorClient after each render/export cycle.
The evaluator scores visual fidelity, topology quality, depth alignment,
and vertex accuracy — returning structured feedback the agent can act on.

Implement EvaluatorBackend to plug in any evaluation service (local model,
remote API, or human-in-the-loop).
"""

from __future__ import annotations

import abc
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from . import secrets as _secrets


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass
class RenderPayload:
    """All artifacts from one Blender render cycle."""
    render_images: list[str]          # paths to rendered PNG/EXR images
    depth_map_path: str | None = None # path to depth EXR/PNG
    ply_path: str | None = None       # path to exported point cloud / mesh
    vertex_positions: list[list[float]] = field(default_factory=list)   # [[x,y,z],…]
    topology_stats: dict[str, Any] = field(default_factory=dict)        # from mcp_connector
    metadata: dict[str, Any] = field(default_factory=dict)              # arbitrary extra info


@dataclass
class EvaluationResult:
    """Scores and actionable feedback from the evaluator."""
    overall_score: float              # 0.0 – 1.0
    visual_fidelity: float = 0.0     # render vs reference comparison
    topology_quality: float = 0.0    # manifold, quads vs tris, watertight
    depth_alignment: float = 0.0     # depth map vs reference depth
    vertex_accuracy: float = 0.0     # vertex positions vs reference mesh
    feedback: list[str] = field(default_factory=list)   # human-readable notes
    raw: dict[str, Any] = field(default_factory=dict)   # full evaluator response


# ------------------------------------------------------------------
# Abstract backend protocol
# ------------------------------------------------------------------

class EvaluatorBackend(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, payload: RenderPayload, reference: dict[str, Any]) -> EvaluationResult:
        ...


# ------------------------------------------------------------------
# HTTP backend — talks to an evaluation REST API
# ------------------------------------------------------------------

class HTTPEvaluatorBackend(EvaluatorBackend):
    """
    POST multipart/form-data to an evaluation endpoint.

    Expected response JSON schema:
    {
        "overall_score": float,
        "visual_fidelity": float,
        "topology_quality": float,
        "depth_alignment": float,
        "vertex_accuracy": float,
        "feedback": [str, ...]
    }
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.headers: dict[str, str] = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout

    def evaluate(self, payload: RenderPayload, reference: dict[str, Any]) -> EvaluationResult:
        files: dict[str, Any] = {}
        data: dict[str, str] = {}

        for i, img_path in enumerate(payload.render_images):
            img_bytes = Path(img_path).read_bytes()
            files[f"render_{i}"] = (Path(img_path).name, img_bytes, "image/png")

        if payload.depth_map_path and Path(payload.depth_map_path).exists():
            files["depth_map"] = (
                Path(payload.depth_map_path).name,
                Path(payload.depth_map_path).read_bytes(),
                "image/png",
            )

        if payload.ply_path and Path(payload.ply_path).exists():
            files["mesh"] = (
                Path(payload.ply_path).name,
                Path(payload.ply_path).read_bytes(),
                "application/octet-stream",
            )

        data["vertex_positions"] = json.dumps(payload.vertex_positions)
        data["topology_stats"] = json.dumps(payload.topology_stats)
        data["reference"] = json.dumps(reference)
        data["metadata"] = json.dumps(payload.metadata)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/evaluate",
                headers=self.headers,
                files=files,
                data=data,
            )
            resp.raise_for_status()
            body = resp.json()

        return EvaluationResult(
            overall_score=body["overall_score"],
            visual_fidelity=body.get("visual_fidelity", 0.0),
            topology_quality=body.get("topology_quality", 0.0),
            depth_alignment=body.get("depth_alignment", 0.0),
            vertex_accuracy=body.get("vertex_accuracy", 0.0),
            feedback=body.get("feedback", []),
            raw=body,
        )


# ------------------------------------------------------------------
# Claude-as-evaluator backend (vision model judges render quality)
# ------------------------------------------------------------------

class ClaudeEvaluatorBackend(EvaluatorBackend):
    """
    Uses Claude's vision to compare renders against a reference image.
    Returns scores based on model critique.
    """

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-6"):
        _secrets.load()
        self.api_key = api_key or _secrets.get("ANTHROPIC_API_KEY")
        self.model = model

    def evaluate(self, payload: RenderPayload, reference: dict[str, Any]) -> EvaluationResult:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)

        content: list[Any] = []

        if "image_path" in reference:
            ref_b64 = base64.standard_b64encode(Path(reference["image_path"]).read_bytes()).decode()
            content.append({
                "type": "text",
                "text": "Reference image (target to match):",
            })
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": ref_b64},
            })

        for img_path in payload.render_images[:4]:   # cap at 4 images
            render_b64 = base64.standard_b64encode(Path(img_path).read_bytes()).decode()
            content.append({"type": "text", "text": f"Rendered output ({Path(img_path).name}):"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": render_b64},
            })

        topo = payload.topology_stats
        content.append({
            "type": "text",
            "text": (
                f"Topology: {topo}\n"
                f"Vertex count: {len(payload.vertex_positions)}\n"
                "Score each dimension 0–1 and list actionable feedback.\n"
                "Respond ONLY with JSON: "
                '{"overall_score":…,"visual_fidelity":…,"topology_quality":…,'
                '"depth_alignment":…,"vertex_accuracy":…,"feedback":[…]}'
            ),
        })

        message = client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": content}],
        )
        body = json.loads(message.content[0].text)
        return EvaluationResult(
            overall_score=body["overall_score"],
            visual_fidelity=body.get("visual_fidelity", 0.0),
            topology_quality=body.get("topology_quality", 0.0),
            depth_alignment=body.get("depth_alignment", 0.0),
            vertex_accuracy=body.get("vertex_accuracy", 0.0),
            feedback=body.get("feedback", []),
            raw=body,
        )


# ------------------------------------------------------------------
# OpenAI vision evaluator backend
# ------------------------------------------------------------------

class OpenAIEvaluatorBackend(EvaluatorBackend):
    """
    Uses GPT-4o vision to compare renders against a reference image.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        openrouter: bool = False,
    ):
        _secrets.load()
        if openrouter:
            self.api_key = api_key or _secrets.get("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.api_key = api_key or _secrets.get("OPENAI_API_KEY")
            self.base_url = None
        self.model = model

    def evaluate(self, payload: RenderPayload, reference: dict[str, Any]) -> EvaluationResult:
        from openai import OpenAI
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)

        content: list[Any] = []

        if "image_path" in reference:
            ref_b64 = base64.standard_b64encode(Path(reference["image_path"]).read_bytes()).decode()
            content.append({"type": "text", "text": "Reference image (target to match):"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}", "detail": "high"},
            })

        for img_path in payload.render_images[:4]:
            render_b64 = base64.standard_b64encode(Path(img_path).read_bytes()).decode()
            content.append({"type": "text", "text": f"Rendered output ({Path(img_path).name}):"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{render_b64}", "detail": "high"},
            })

        topo = payload.topology_stats
        content.append({
            "type": "text",
            "text": (
                f"Topology: {topo}\n"
                f"Vertex count: {len(payload.vertex_positions)}\n"
                "Score each dimension 0–1 and list actionable feedback.\n"
                "Respond ONLY with JSON: "
                '{"overall_score":…,"visual_fidelity":…,"topology_quality":…,'
                '"depth_alignment":…,"vertex_accuracy":…,"feedback":[…]}'
            ),
        })

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}],
        )
        body = json.loads(response.choices[0].message.content)
        return EvaluationResult(
            overall_score=body["overall_score"],
            visual_fidelity=body.get("visual_fidelity", 0.0),
            topology_quality=body.get("topology_quality", 0.0),
            depth_alignment=body.get("depth_alignment", 0.0),
            vertex_accuracy=body.get("vertex_accuracy", 0.0),
            feedback=body.get("feedback", []),
            raw=body,
        )


# ------------------------------------------------------------------
# Codex evaluator backend (uses imagegen CodexBackend.judge)
# ------------------------------------------------------------------

_EVAL_PROMPT = """\
You are a 3-D render quality evaluator.
{reference_note}
The remaining image(s) are renders of a 3-D model to evaluate.
Topology stats: {topo}
Vertex count: {verts}

Score each dimension from 0.0 to 1.0 and list up to 5 short actionable feedback items.
"""

_EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_score":    {"type": "number"},
        "visual_fidelity":  {"type": "number"},
        "topology_quality": {"type": "number"},
        "depth_alignment":  {"type": "number"},
        "vertex_accuracy":  {"type": "number"},
        "feedback":         {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overall_score", "visual_fidelity", "topology_quality",
                 "depth_alignment", "vertex_accuracy", "feedback"],
    "additionalProperties": False,
}


class CodexEvaluatorBackend(EvaluatorBackend):
    """Uses the Codex CLI (ChatGPT subscription) vision judge to score renders."""

    def __init__(self, model: str | None = None):
        from imagegen.backends import CodexBackend
        self._codex = CodexBackend(model=model)

    def evaluate(self, payload: RenderPayload, reference: dict[str, Any]) -> EvaluationResult:
        images: list[Any] = []
        reference_note = ""

        if "image_path" in reference:
            images.append(reference["image_path"])
            reference_note = "Image 1 is the reference target. "

        for img_path in payload.render_images[:4]:
            images.append(img_path)

        prompt = _EVAL_PROMPT.format(
            reference_note=reference_note,
            topo=payload.topology_stats or "unknown",
            verts=len(payload.vertex_positions),
        )

        body = self._codex.judge(prompt, images, _EVAL_SCHEMA)
        return EvaluationResult(
            overall_score=body["overall_score"],
            visual_fidelity=body.get("visual_fidelity", 0.0),
            topology_quality=body.get("topology_quality", 0.0),
            depth_alignment=body.get("depth_alignment", 0.0),
            vertex_accuracy=body.get("vertex_accuracy", 0.0),
            feedback=body.get("feedback", []),
            raw=body,
        )


# ------------------------------------------------------------------
# Client façade (used by BlenderAgent)
# ------------------------------------------------------------------

class EvaluatorClient:
    """
    Thin façade the agent calls.  Swap backends without touching agent code.
    """

    def __init__(self, backend: EvaluatorBackend):
        self.backend = backend

    def evaluate(self, payload: RenderPayload, reference: dict[str, Any]) -> EvaluationResult:
        return self.backend.evaluate(payload, reference)

    # Convenience factory methods

    @classmethod
    def from_http(cls, base_url: str, api_key: str = "") -> "EvaluatorClient":
        return cls(HTTPEvaluatorBackend(base_url, api_key))

    @classmethod
    def from_claude(cls, api_key: str = "", model: str = "claude-sonnet-4-6") -> "EvaluatorClient":
        return cls(ClaudeEvaluatorBackend(api_key, model))

    @classmethod
    def from_openai(cls, api_key: str = "", model: str = "gpt-4o") -> "EvaluatorClient":
        return cls(OpenAIEvaluatorBackend(api_key, model, openrouter=False))

    @classmethod
    def from_openrouter(cls, api_key: str = "", model: str = "openai/gpt-4o") -> "EvaluatorClient":
        return cls(OpenAIEvaluatorBackend(api_key, model, openrouter=True))

    @classmethod
    def from_codex(cls, model: str | None = None) -> "EvaluatorClient":
        return cls(CodexEvaluatorBackend(model))
