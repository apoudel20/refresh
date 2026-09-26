"""Image-model backends: the Codex CLI (your ChatGPT/Codex subscription) and OpenRouter.

Both expose the same two calls, so the pipeline doesn't care which one runs:

* ``generate(prompt, references=..., aspect_ratio=..., n=...) -> GenerationResult``
* ``judge(prompt, images, schema) -> dict`` — a vision model answering with JSON
  (used by ``atlas review`` to check cells against their descriptions).

Codex backend
    Shells out to ``codex exec`` non-interactively. Codex's built-in image tool
    writes its results to ``~/.codex/generated_images/<thread_id>/``; we read
    the thread id from the ``--json`` event stream and collect the files.
    Reference images are attached with ``-i``. Billing goes to the ChatGPT plan
    Codex is logged into (``codex login status``).

OpenRouter backend
    ``POST /api/v1/images`` (reference images via ``input_references``) and
    ``POST /api/v1/chat/completions`` with a JSON schema for judging.
    Needs ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import httpx
from dotenv import load_dotenv
from PIL import Image

from imagegen.ops import ImageLike, image_to_url, load_image, nearest_aspect

load_dotenv()

DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-flash-image"
DEFAULT_OPENROUTER_JUDGE = "google/gemini-3.8-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

COMMON_ASPECTS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


class ImageGenError(RuntimeError):
    """A backend failed or returned something unusable."""

    def __init__(self, message: str, detail: Any = None):
        super().__init__(message)
        self.detail = detail


@dataclass
class GenerationResult:
    images: list[Image.Image]
    backend: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def image(self) -> Image.Image:
        return self.images[0]


class Backend(Protocol):
    name: str
    model: str
    aspect_ratios: list[str]

    def generate(
        self,
        prompt: str,
        *,
        references: Sequence[ImageLike] = (),
        aspect_ratio: str | None = None,
        n: int = 1,
        **options: Any,
    ) -> GenerationResult: ...

    def judge(self, prompt: str, images: Sequence[ImageLike], schema: dict[str, Any]) -> dict[str, Any]: ...


def _as_pil(src: ImageLike) -> Image.Image:
    return load_image(src)


# --------------------------------------------------------------------------- #
# Codex CLI
# --------------------------------------------------------------------------- #

_CODEX_GENERATE = """\
You are running as a non-interactive image generation backend inside a script.
Call your image generation tool exactly {n} time(s). Do not run shell commands, do not
read or write files, do not ask questions, and do not describe the image.
{refs}{aspect}
Image request:
<<<
{prompt}
>>>

When the image generation tool has finished, reply with only: DONE
"""

_CODEX_JUDGE = """\
You are running as a non-interactive vision reviewer inside a script.
Do NOT generate or edit images. Do not run shell commands and do not read files.
Look carefully at the attached image(s) and answer in the required JSON format.

{prompt}
"""


class CodexBackend:
    """Image generation through the local ``codex`` CLI and its logged-in subscription."""

    name = "codex"
    # Codex's image tool renders square, landscape (3:2) or portrait (2:3) canvases.
    aspect_ratios = ["1:1", "3:2", "2:3"]

    def __init__(
        self,
        model: str | None = None,
        *,
        codex_bin: str | None = None,
        reasoning_effort: str = "low",
        timeout: float = 900.0,
        images_root: str | Path | None = None,
        isolated: bool = True,
        extra_args: Sequence[str] = (),
        max_retries: int = 2,
    ):
        self.model = model or os.getenv("IMAGEGEN_CODEX_MODEL") or DEFAULT_CODEX_MODEL
        self.max_retries = max_retries
        self.codex_bin = codex_bin or os.getenv("IMAGEGEN_CODEX_BIN") or shutil.which("codex")
        if not self.codex_bin:
            raise ImageGenError(
                "codex CLI not found on PATH. Install it (https://github.com/openai/codex) and run "
                "`codex login`, or use --backend openrouter."
            )
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        codex_home = Path(os.getenv("CODEX_HOME", Path.home() / ".codex"))
        self.images_root = Path(images_root) if images_root else codex_home / "generated_images"
        # isolated: skip the user's config.toml (MCP servers, rules, profiles) for a fast, predictable
        # sub-agent. Auth still comes from CODEX_HOME.
        self.isolated = isolated
        self.extra_args = list(extra_args)

    # -- public ------------------------------------------------------------- #

    def generate(
        self,
        prompt: str,
        *,
        references: Sequence[ImageLike] = (),
        aspect_ratio: str | None = None,
        n: int = 1,
        **options: Any,
    ) -> GenerationResult:
        refs = ""
        if references:
            refs = (
                f"{len(references)} reference image(s) are attached; the request below refers to them "
                "as image 1, image 2, ... in attachment order.\n"
            )
        aspect = ""
        if aspect_ratio:
            shape = {"1:1": "square", "3:2": "landscape", "2:3": "portrait"}.get(aspect_ratio, "")
            aspect = f"Output aspect ratio: {aspect_ratio}{f' ({shape})' if shape else ''}.\n"
        text = _CODEX_GENERATE.format(n=n, refs=refs, aspect=aspect, prompt=prompt.strip())
        run = self._exec(text, references)
        files = sorted(
            (p for p in (self.images_root / run["thread_id"]).glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")),
            key=lambda p: p.stat().st_mtime,
        )
        if not files:
            raise ImageGenError(
                f"Codex finished without producing an image (thread {run['thread_id']}). "
                f"Last message: {run['message']!r}",
                detail=run,
            )
        return GenerationResult(
            images=[load_image(p) for p in files],
            backend=self.name,
            model=self.model,
            usage=run["usage"],
            meta={"thread_id": run["thread_id"], "files": [str(p) for p in files], "seconds": run["seconds"]},
        )

    def judge(self, prompt: str, images: Sequence[ImageLike], schema: dict[str, Any]) -> dict[str, Any]:
        run = self._exec(_CODEX_JUDGE.format(prompt=prompt.strip()), images, output_schema=schema)
        return _parse_json(run["message"])

    # -- internals ---------------------------------------------------------- #

    # Failures worth retrying: network drops, overload, rate limits. Not: bad model, auth, bad args.
    _TRANSIENT = ("stream disconnected", "error sending request", "timed out", "rate limit", "429", "500", "502",
                  "503", "504", "overloaded", "connection reset", "temporarily unavailable")  # fmt: skip

    def _exec(
        self, prompt: str, images: Sequence[ImageLike], output_schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                return self._exec_once(prompt, images, output_schema)
            except ImageGenError as exc:
                text = str(exc).lower()
                if attempt == self.max_retries or not any(t in text for t in self._TRANSIENT):
                    raise
                time.sleep(min(5 * 2**attempt, 30))
        raise AssertionError("unreachable")

    def _exec_once(
        self, prompt: str, images: Sequence[ImageLike], output_schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="imagegen-codex-") as tmp:
            work = Path(tmp)
            cmd = [
                self.codex_bin, "exec", "--json", "--skip-git-repo-check", "--ephemeral",
                "-s", "read-only", "-m", self.model,
                "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
                "-C", str(work),
                "-o", str(work / "last_message.txt"),
            ]  # fmt: skip
            if self.isolated:
                cmd += ["--ignore-user-config", "--ignore-rules"]
            if output_schema is not None:
                (work / "schema.json").write_text(json.dumps(output_schema))
                cmd += ["--output-schema", str(work / "schema.json")]
            cmd += self.extra_args
            # Prompt goes on stdin ("-"), placed before -i because -i accepts several values.
            cmd.append("-")
            for i, im in enumerate(images):
                path = work / f"ref_{i + 1}.png"
                _as_pil(im).save(path)
                cmd += ["-i", str(path)]

            start = time.monotonic()
            try:
                proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise ImageGenError(f"codex exec timed out after {self.timeout:.0f}s") from exc
            seconds = round(time.monotonic() - start, 1)

            thread_id, message, usage, failure = None, "", {}, None
            for line in proc.stdout.splitlines():
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                kind = ev.get("type")
                if kind == "thread.started":
                    thread_id = ev.get("thread_id")
                elif kind == "item.completed" and ev.get("item", {}).get("type") == "agent_message":
                    message = ev["item"].get("text", "")
                elif kind == "turn.completed":
                    usage = ev.get("usage") or {}
                elif kind == "turn.failed":
                    failure = (ev.get("error") or {}).get("message", "turn failed")
            last = work / "last_message.txt"
            if last.is_file() and last.read_text().strip():
                message = last.read_text().strip()

        if failure or proc.returncode != 0 or not thread_id:
            detail = failure or proc.stderr.strip()[-2000:] or proc.stdout.strip()[-2000:]
            hint = ""
            if detail and "requires a newer version of Codex" in detail:
                hint = " (run `codex update`, or pick an older model with --model / IMAGEGEN_CODEX_MODEL)"
            raise ImageGenError(f"codex exec failed{hint}: {_short(detail)}", detail=detail)
        return {"thread_id": thread_id, "message": message, "usage": usage, "seconds": seconds}


# --------------------------------------------------------------------------- #
# OpenRouter
# --------------------------------------------------------------------------- #


class OpenRouterBackend:
    """``/api/v1/images`` for generation, ``/api/v1/chat/completions`` for judging."""

    name = "openrouter"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        judge_model: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: float = 300.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
        app_name: str = "imagegen",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ImageGenError(
                "OPENROUTER_API_KEY is not set. Export it or put it in .env, or use --backend codex."
            )
        self.model = model or os.getenv("IMAGEGEN_OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
        self.judge_model = judge_model or os.getenv("IMAGEGEN_JUDGE_MODEL") or DEFAULT_OPENROUTER_JUDGE
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": app_name,
        }
        self._caps: dict[str, dict[str, Any]] = {}

    # -- discovery ---------------------------------------------------------- #

    def image_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/images/models").get("data", [])

    def capabilities(self, model: str | None = None) -> dict[str, Any]:
        """``supported_parameters`` for a model ({} if discovery fails — we then send params as-is)."""
        model = model or self.model
        if model not in self._caps:
            try:
                found = next((m for m in self.image_models() if m.get("id") == model), None)
                self._caps[model] = (found or {}).get("supported_parameters") or {}
            except ImageGenError:
                self._caps[model] = {}
        return self._caps[model]

    @property
    def aspect_ratios(self) -> list[str]:
        values = (self.capabilities().get("aspect_ratio") or {}).get("values")
        return [v for v in values if v != "auto"] if values else COMMON_ASPECTS

    # -- public ------------------------------------------------------------- #

    def generate(
        self,
        prompt: str,
        *,
        references: Sequence[ImageLike] = (),
        aspect_ratio: str | None = None,
        n: int = 1,
        resolution: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        seed: int | None = None,
        **options: Any,
    ) -> GenerationResult:
        caps = self.capabilities()
        body: dict[str, Any] = {"model": self.model, "prompt": prompt}
        if references:
            body["input_references"] = [
                {"type": "image_url", "image_url": {"url": image_to_url(_as_pil(r))}} for r in references
            ]
        if aspect_ratio:
            body["aspect_ratio"] = (
                aspect_ratio if aspect_ratio in self.aspect_ratios else nearest_aspect(*_ratio_pair(aspect_ratio), self.aspect_ratios)
            )
        for key, val in (("n", n if n != 1 else None), ("resolution", resolution), ("quality", quality),
                         ("background", background), ("seed", seed)):  # fmt: skip
            if val is not None and (not caps or key in caps):
                body[key] = val
        body.update({k: v for k, v in options.items() if v is not None})

        payload = self._request("POST", "/images", body)
        images = []
        for item in payload.get("data") or []:
            if item.get("b64_json"):
                images.append(load_image(base64.b64decode(item["b64_json"])))
            elif item.get("url"):
                images.append(load_image(self._client.get(item["url"]).content))
        if not images:
            raise ImageGenError("OpenRouter returned no images", detail=payload)
        return GenerationResult(
            images=images,
            backend=self.name,
            model=self.model,
            usage=payload.get("usage") or {},
            meta={"request": {k: v for k, v in body.items() if k not in ("prompt", "input_references")}},
        )

    def judge(self, prompt: str, images: Sequence[ImageLike], schema: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": image_to_url(_as_pil(im))}} for im in images]
        body = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "review", "strict": True, "schema": schema}},
        }
        payload = self._request("POST", "/chat/completions", body)
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ImageGenError("Unexpected chat completion payload", detail=payload) from exc
        return _parse_json(text)

    # -- http --------------------------------------------------------------- #

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.request(method, url, json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                last_exc = exc
                time.sleep(min(2**attempt, 8))
                continue
            # 502 = generation failed upstream (not billed), safe to retry.
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
                raise ImageGenError(f"OpenRouter {path} failed ({resp.status_code}): {msg}", detail=err)
            try:
                return resp.json()
            except ValueError as exc:
                raise ImageGenError(f"OpenRouter {path} returned non-JSON", detail=resp.text) from exc
        raise ImageGenError(f"OpenRouter {path} failed after {self.max_retries} attempts: {last_exc}")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

class OpenAIBackend:
    """DALL-E 3 via the OpenAI API. Needs OPENAI_API_KEY."""

    name = "openai"
    aspect_ratios = ["1:1", "16:9", "9:16"]

    def __init__(self, model: str | None = None, *, api_key: str | None = None, timeout: float = 120.0):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ImageGenError("OPENAI_API_KEY is not set.")
        self.model = model or os.getenv("IMAGEGEN_OPENAI_MODEL") or "dall-e-3"
        self._timeout = timeout

    @property
    def _client(self):
        import openai
        return openai.OpenAI(api_key=self.api_key, timeout=self._timeout)

    def generate(
        self,
        prompt: str,
        *,
        references: Sequence[ImageLike] = (),
        aspect_ratio: str | None = None,
        n: int = 1,
        **options: Any,
    ) -> GenerationResult:
        size_map = {"16:9": "1792x1024", "9:16": "1024x1792"}
        size = size_map.get(aspect_ratio or "1:1", "1024x1024")
        resp = self._client.images.generate(model=self.model, prompt=prompt, n=1, size=size)
        img_bytes = httpx.get(resp.data[0].url, timeout=60).content
        img = load_image(img_bytes)
        return GenerationResult(images=[img], backend=self.name, model=self.model)

    def judge(self, prompt: str, images: Sequence[ImageLike], schema: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": image_to_url(_as_pil(im))}} for im in images]
        resp = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_schema", "json_schema": {"name": "review", "strict": True, "schema": schema}},
            max_tokens=512,
        )
        return _parse_json(resp.choices[0].message.content)


BACKENDS = {"codex": CodexBackend, "openrouter": OpenRouterBackend, "openai": OpenAIBackend}


def get_backend(name: str | None = None, model: str | None = None, **kwargs: Any) -> Backend:
    """``name`` or $IMAGEGEN_BACKEND; otherwise codex if installed, else openrouter."""
    name = name or os.getenv("IMAGEGEN_BACKEND")
    if not name:
        name = "codex" if (os.getenv("IMAGEGEN_CODEX_BIN") or shutil.which("codex")) else "openrouter"
    if name not in BACKENDS:
        raise ImageGenError(f"Unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
    return BACKENDS[name](model=model, **kwargs)


def backend_status() -> dict[str, Any]:
    """What's usable right now, without spending anything."""
    codex_bin = os.getenv("IMAGEGEN_CODEX_BIN") or shutil.which("codex")
    codex: dict[str, Any] = {"installed": bool(codex_bin), "model": os.getenv("IMAGEGEN_CODEX_MODEL") or DEFAULT_CODEX_MODEL}
    if codex_bin:
        for key, args in (("version", ["--version"]), ("login", ["login", "status"])):
            try:
                out = subprocess.run([codex_bin, *args], capture_output=True, text=True, timeout=20)
                codex[key] = (out.stdout or out.stderr).strip().splitlines()[-1] if (out.stdout or out.stderr).strip() else ""
            except (OSError, subprocess.TimeoutExpired) as exc:
                codex[key] = f"error: {exc}"
    return {
        "default": os.getenv("IMAGEGEN_BACKEND") or ("codex" if codex_bin else "openrouter"),
        "codex": codex,
        "openrouter": {
            "api_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
            "model": os.getenv("IMAGEGEN_OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
            "judge_model": os.getenv("IMAGEGEN_JUDGE_MODEL") or DEFAULT_OPENROUTER_JUDGE,
        },
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _ratio_pair(aspect: str) -> tuple[float, float]:
    w, h = aspect.split(":")
    return float(w), float(h)


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                pass
    raise ImageGenError("Reviewer did not return valid JSON", detail=text)


def _short(text: str, limit: int = 400) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"

