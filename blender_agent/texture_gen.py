"""Texture generation via the imagegen pipeline (imagegen-skill).

TextureGenerator wraps imagegen.pipeline.generate / edit so the agent
can call generate_from_prompt and generate_from_reference without
knowing which backend is active.
"""

from __future__ import annotations

from dataclasses import dataclass

from imagegen import ops, pipeline
from imagegen.backends import Backend, get_backend


@dataclass
class TextureGenConfig:
    backend: str | None = None          # "codex" | "openrouter" | None → auto-detect
    model: str | None = None            # override default model for the backend
    default_size: tuple[int, int] = (1024, 1024)


_TYPE_STYLE: dict[str, str] = {
    "albedo":    "seamless tileable PBR albedo texture, uniform lighting, no shadows, diffuse colour only",
    "normal":    "seamless tileable PBR normal map, purple-blue tones, OpenGL convention",
    "roughness": "seamless tileable PBR roughness map, greyscale, uniform lighting",
}
_DEFAULT_STYLE = _TYPE_STYLE["albedo"]


class TextureGenerator:
    def __init__(self, config: TextureGenConfig | None = None):
        self._cfg = config or TextureGenConfig()
        self._backend: Backend | None = None

    def _get_backend(self) -> Backend:
        if self._backend is None:
            self._backend = get_backend(self._cfg.backend, self._cfg.model)
        return self._backend

    def generate_from_prompt(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = "",      # kept for API compat; not used by imagegen pipeline
        size: tuple[int, int] | int | None = None,
        texture_type: str = "albedo",
    ) -> str:
        style = _TYPE_STYLE.get(texture_type, _DEFAULT_STYLE)
        # Accept int (square) or tuple from tool schema
        if isinstance(size, int):
            size = (size, size)
        images, _ = pipeline.generate(
            f"{prompt}. {style}",
            backend=self._get_backend(),
            size=size or self._cfg.default_size,
        )
        return str(ops.save_image(images[0], output_path))

    def generate_from_reference(
        self,
        reference_image_path: str,
        output_path: str,
        prompt: str = "",
        strength: float = 0.6,          # kept for API compat; shapes instruction phrasing
        texture_type: str = "albedo",
    ) -> str:
        style = _TYPE_STYLE.get(texture_type, _DEFAULT_STYLE)
        base = f"{prompt}. {style}" if prompt else style
        if strength < 0.4:
            instruction = f"Lightly adjust while mostly preserving the original: {base}"
        elif strength > 0.7:
            instruction = f"Completely retexture: {base}"
        else:
            instruction = base
        result = pipeline.edit(
            reference_image_path,
            instruction,
            backend=self._get_backend(),
        )
        return str(ops.save_image(result.image, output_path))
