"""
Texture generation using image generation APIs.
Supports Stability AI, fal.ai, and Replicate as backends.
"""

import base64
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from . import secrets as _secrets


Backend = Literal["stability", "fal", "replicate"]


@dataclass
class TextureGenConfig:
    backend: Backend = "stability"
    api_key: str = field(default_factory=lambda: os.environ.get("STABILITY_API_KEY", ""))
    fal_key: str = field(default_factory=lambda: os.environ.get("FAL_KEY", ""))
    replicate_key: str = field(default_factory=lambda: os.environ.get("REPLICATE_API_TOKEN", ""))
    default_size: int = 1024
    default_steps: int = 30


class TextureGenerator:
    """
    Generates PBR-ready texture images from text prompts or reference images.
    Output is always a PNG saved to disk; returns the path.
    """

    def __init__(self, config: TextureGenConfig | None = None):
        _secrets.load()
        self.cfg = config or TextureGenConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_prompt(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = "seam, low quality, blurry",
        size: int | None = None,
        texture_type: str = "albedo",  # albedo | normal | roughness | metallic
    ) -> str:
        """Generate a texture image from a text description. Returns saved path."""
        full_prompt = self._enrich_prompt(prompt, texture_type)
        size = size or self.cfg.default_size

        if self.cfg.backend == "stability":
            image_bytes = self._stability_generate(full_prompt, negative_prompt, size)
        elif self.cfg.backend == "fal":
            image_bytes = self._fal_generate(full_prompt, negative_prompt, size)
        elif self.cfg.backend == "replicate":
            image_bytes = self._replicate_generate(full_prompt, negative_prompt, size)
        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_bytes)
        return str(path)

    def generate_from_reference(
        self,
        reference_image_path: str,
        output_path: str,
        prompt: str = "",
        strength: float = 0.6,
        texture_type: str = "albedo",
    ) -> str:
        """
        Image-to-image texture generation: use a reference photo as style seed.
        Returns saved path.
        """
        full_prompt = self._enrich_prompt(prompt or "seamless tileable texture", texture_type)
        ref_bytes = Path(reference_image_path).read_bytes()

        if self.cfg.backend == "stability":
            image_bytes = self._stability_img2img(full_prompt, ref_bytes, strength)
        elif self.cfg.backend == "fal":
            image_bytes = self._fal_img2img(full_prompt, ref_bytes, strength)
        else:
            raise NotImplementedError(f"img2img not implemented for backend: {self.cfg.backend}")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_bytes)
        return str(path)

    def generate_normal_map(self, albedo_path: str, output_path: str) -> str:
        """
        Derive a normal map from an existing albedo texture via img2img with
        a normal-map-specific prompt + negative guidance.
        """
        return self.generate_from_reference(
            albedo_path,
            output_path,
            prompt="normal map, blue purple surface normals, flat lighting",
            strength=0.45,
            texture_type="normal",
        )

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _stability_generate(self, prompt: str, negative_prompt: str, size: int) -> bytes:
        url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {self.cfg.api_key}", "Accept": "image/*"},
                data={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "output_format": "png",
                    "width": size,
                    "height": size,
                },
            )
            resp.raise_for_status()
            return resp.content

    def _stability_img2img(self, prompt: str, image_bytes: bytes, strength: float) -> bytes:
        url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {self.cfg.api_key}", "Accept": "image/*"},
                data={"prompt": prompt, "mode": "image-to-image", "strength": strength, "output_format": "png"},
                files={"image": ("reference.png", image_bytes, "image/png")},
            )
            resp.raise_for_status()
            return resp.content

    def _fal_generate(self, prompt: str, negative_prompt: str, size: int) -> bytes:
        url = "https://fal.run/fal-ai/flux/dev"
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Key {self.cfg.fal_key}"},
                json={"prompt": prompt, "negative_prompt": negative_prompt, "image_size": f"{size}x{size}"},
            )
            resp.raise_for_status()
            image_url = resp.json()["images"][0]["url"]
            return client.get(image_url).content

    def _fal_img2img(self, prompt: str, image_bytes: bytes, strength: float) -> bytes:
        b64 = base64.b64encode(image_bytes).decode()
        url = "https://fal.run/fal-ai/flux/dev/image-to-image"
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Key {self.cfg.fal_key}"},
                json={"prompt": prompt, "image_url": f"data:image/png;base64,{b64}", "strength": strength},
            )
            resp.raise_for_status()
            image_url = resp.json()["images"][0]["url"]
            return client.get(image_url).content

    def _replicate_generate(self, prompt: str, negative_prompt: str, size: int) -> bytes:
        import time
        url = "https://api.replicate.com/v1/models/black-forest-labs/flux-dev/predictions"
        headers = {"Authorization": f"Bearer {self.cfg.replicate_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=180) as client:
            resp = client.post(url, headers=headers, json={"input": {"prompt": prompt, "width": size, "height": size}})
            resp.raise_for_status()
            prediction = resp.json()
            poll_url = prediction["urls"]["get"]
            for _ in range(60):
                time.sleep(3)
                poll = client.get(poll_url, headers=headers).json()
                if poll["status"] == "succeeded":
                    return client.get(poll["output"][0]).content
                if poll["status"] == "failed":
                    raise RuntimeError(f"Replicate prediction failed: {poll.get('error')}")
            raise TimeoutError("Replicate prediction timed out")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_prompt(prompt: str, texture_type: str) -> str:
        suffixes = {
            "albedo": "seamless tileable albedo texture, diffuse color map, no shadows, flat lighting, 4K",
            "normal": "seamless tileable normal map, blue purple tangent space normals, flat",
            "roughness": "seamless tileable roughness map, grayscale, PBR",
            "metallic": "seamless tileable metallic map, grayscale, PBR",
        }
        return f"{prompt}, {suffixes.get(texture_type, 'seamless tileable texture, 4K')}"
