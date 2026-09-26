"""Minimal OpenRouter embeddings client (images and text).

Adapted from ``harness/embeddings.py`` in the mongodb-hack-harness repo, trimmed to what
render-eval uses, so this package has no dependency on that repo.

OpenRouter exposes an OpenAI-compatible ``POST /api/v1/embeddings`` endpoint. Multimodal
models accept an ``input`` list of ``{"content": [...]}`` blocks with ``image_url`` parts
(http(s) URLs or base64 data URLs) or ``text`` parts.

The API key comes from ``OPENROUTER_API_KEY``, or from a ``.env`` file found by walking up
from the directory you run in.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Union

import httpx
import numpy as np
from dotenv import find_dotenv, load_dotenv
from PIL import Image

load_dotenv(find_dotenv(usecwd=True))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "voyageai/voyage-multimodal-3.5"  # Voyage AI is MongoDB's embedding team; 1024 dims

ImageSource = Union[str, Path, bytes, Image.Image]


class EmbeddingError(RuntimeError):
    """Raised when OpenRouter returns an error or an unexpected payload."""

    def __init__(self, message: str, status_code: int | None = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class EmbeddingResult:
    """One request's vectors, in input order."""

    vectors: list[list[float]]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0

    def __getitem__(self, i: int) -> list[float]:
        return self.vectors[i]

    def __len__(self) -> int:
        return len(self.vectors)


def _data_url(data: bytes, hint: str | None = None) -> str:
    mime = "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    elif not data.startswith(b"\x89PNG") and hint:
        guessed, _ = mimetypes.guess_type(hint)
        mime = guessed if guessed and guessed.startswith("image/") else mime
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def image_to_url(src: ImageSource) -> str:
    """A URL OpenRouter accepts in ``image_url.url``: http(s)/data URLs pass through, the rest is base64-encoded."""
    if isinstance(src, Image.Image):
        buf = io.BytesIO()
        src.save(buf, format="PNG")
        return _data_url(buf.getvalue(), "x.png")
    if isinstance(src, (bytes, bytearray)):
        return _data_url(bytes(src))
    if isinstance(src, str) and src.startswith(("http://", "https://", "data:")):
        return src
    p = Path(src).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {src}")
    return _data_url(p.read_bytes(), p.name)


class OpenRouterEmbedder:
    """Thin client for OpenRouter's ``/embeddings`` endpoint with image and text inputs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        dimensions: int | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EmbeddingError(
                "OPENROUTER_API_KEY is not set. Export it or put it in a .env file in your project. "
                "Get a key at https://openrouter.ai/keys"
            )
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "render-eval",
        }

    def embed_inputs(self, inputs: Sequence[dict[str, Any]]) -> EmbeddingResult:
        """POST pre-built ``{"content": [...]}`` blocks; returns vectors in input order."""
        if not inputs:
            raise ValueError("inputs must be non-empty")
        body: dict[str, Any] = {"model": self.model, "input": list(inputs), "encoding_format": "float"}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        payload = self._post(body)
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(inputs):
            got = len(data) if isinstance(data, list) else data
            raise EmbeddingError(f"Expected {len(inputs)} embeddings, got {got!r}", body=payload)
        ordered = sorted(data, key=lambda d: d.get("index", 0))
        return EmbeddingResult(
            vectors=[[float(x) for x in d["embedding"]] for d in ordered],
            model=payload.get("model", self.model),
            usage=payload.get("usage") or {},
        )

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/embeddings"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(url, json=body, headers=self._headers)
            except httpx.HTTPError as exc:  # network-level failure: back off and retry
                last_exc = exc
                time.sleep(min(2**attempt, 8))
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                retry_after = resp.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 8))
                continue
            if resp.status_code >= 400:
                try:
                    err: Any = resp.json()
                except ValueError:
                    err = resp.text
                msg = err.get("error", {}).get("message") if isinstance(err, dict) else err
                raise EmbeddingError(f"OpenRouter embeddings request failed ({resp.status_code}): {msg}",
                                     status_code=resp.status_code, body=err)
            try:
                return resp.json()
            except ValueError as exc:
                raise EmbeddingError("OpenRouter returned non-JSON body", resp.status_code, resp.text) from exc
        raise EmbeddingError(f"OpenRouter request failed after {self.max_retries} attempts: {last_exc}")

    def embed_images(self, images: Sequence[ImageSource]) -> EmbeddingResult:
        """Embed several images in one request."""
        return self.embed_inputs([{"content": [{"type": "image_url", "image_url": {"url": image_to_url(im)}}]}
                                  for im in images])

    def embed_text(self, text: str) -> list[float]:
        return self.embed_inputs([{"content": [{"type": "text", "text": text}]}])[0]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either vector is all zeros."""
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if va.shape != vb.shape:
        raise ValueError(f"Vector length mismatch: {va.shape[0]} vs {vb.shape[0]}")
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
