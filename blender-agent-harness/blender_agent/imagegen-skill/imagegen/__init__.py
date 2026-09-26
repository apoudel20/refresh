"""imagegen: generate, edit (whole or region), upscale and atlas images with AI image models.

Backends: the Codex CLI (your ChatGPT/Codex subscription) or OpenRouter's Image API.
See imagegen/README.md.
"""

from imagegen.atlas import AtlasProject, AtlasSpec, CellSpec
from imagegen.backends import (
    CodexBackend,
    GenerationResult,
    ImageGenError,
    OpenRouterBackend,
    backend_status,
    get_backend,
)
from imagegen.pipeline import EditResult, edit, generate, seamless, upscale

__all__ = [
    "AtlasProject",
    "AtlasSpec",
    "CellSpec",
    "CodexBackend",
    "EditResult",
    "GenerationResult",
    "ImageGenError",
    "OpenRouterBackend",
    "backend_status",
    "edit",
    "generate",
    "get_backend",
    "seamless",
    "upscale",
]
