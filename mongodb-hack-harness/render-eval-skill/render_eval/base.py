"""Shared types for the reference-vs-render evals."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from render_eval.openrouter import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL

DEFAULT_JUDGE_MODEL = os.getenv("RENDER_EVAL_JUDGE_MODEL") or os.getenv("HARNESS_JUDGE_MODEL") or "anthropic/claude-opus-5.5"


@dataclass
class EvalConfig:
    """Knobs shared by all evals. Defaults are chosen to work out of the box."""

    # Preprocessing (see render_eval.pair)
    size: int = 512  # working resolution; both images become size x size
    align: str = "bbox"  # "bbox": crop each image to its foreground box; "none": just pad + resize
    background: str = "neutral"  # "neutral": composite foreground onto bg_color; "keep": keep original pixels
    bg_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    margin: float = 0.08  # padding around the foreground box, as a fraction of its longest side
    mask_model: str = "birefnet-general-lite"  # rembg model used when an image has no alpha channel

    # Model choices
    device: str | None = None  # None -> mps / cuda / cpu, whichever is available
    lpips_net: str = "alex"
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    normals_backend: str = "marigold"  # "marigold" (diffusion normal estimator) or "depth" (from depth gradients)
    normals_model: str = "prs-eth/marigold-normals-v1-1"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int | None = None
    judge_model: str = DEFAULT_JUDGE_MODEL

    # Output
    keep_vectors: bool = False  # include raw embedding vectors in the embedding eval's details
    debug: bool = False  # produce visual artifacts (written by the CLI with --debug-dir)


@dataclass
class EvalResult:
    """Outcome of one eval.

    ``score`` is normalised to [0, 1] with higher meaning "render matches reference better",
    so scores from different evals can be averaged. ``metrics`` holds the raw numbers.
    ``score`` is None when the eval was skipped (for example, no API key) or failed.
    """

    name: str
    score: float | None
    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    skipped: str | None = None
    error: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.score is not None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "score": self.score,
            "metrics": self.metrics,
            "seconds": round(self.seconds, 3),
        }
        if self.details:
            out["details"] = self.details
        if self.skipped:
            out["skipped"] = self.skipped
        if self.error:
            out["error"] = self.error
        return out


class EvalSkipped(Exception):
    """Raise inside an eval to mark it skipped (missing API key, missing input, ...)."""


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
