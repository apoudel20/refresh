"""Eval 8: vision-language-model judge via OpenRouter.

A multimodal LLM sees the reference and the render side by side and grades the
render with a fixed rubric. Calls go through OpenRouter's OpenAI-compatible chat
API using the ``openai`` SDK, so any vision model on OpenRouter works
(``--judge-model``; default ``anthropic/claude-opus-5.5``, or set
``RENDER_EVAL_JUDGE_MODEL``).

Two modes:

* **Rubric** (:func:`evaluate`): absolute 1-10 scores for shape, proportions,
  parts, color and materials, plus missing/extra parts and the top fixes to make.
  score = (mean of the five criteria - 1) / 9.
* **Pairwise** (:func:`compare`): "which of two renders is closer to the
  reference?", asked twice with the order swapped to cancel position bias.
  More reliable than absolute scores for ranking candidates.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

from dotenv import find_dotenv, load_dotenv
from PIL import Image

from render_eval.base import EvalConfig, EvalResult, EvalSkipped, clip01
from render_eval.pair import ImageInput, ImagePair, make_pair

NAME = "judge"
DESCRIPTION = "Vision LLM on OpenRouter grades the render against the reference with a rubric"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CRITERIA = ("shape", "proportions", "parts", "color", "materials")

RUBRIC = """You grade how faithfully a 3D model, rendered in Blender, reproduces the object in a reference image.
Both images were cropped to the object and put on a neutral grey background. Judge the object itself.
Ignore the background, render quality, lighting direction, noise and small camera-angle differences,
unless they hide the object's form.

Score each criterion from 1 to 10, and write the reason before the score:
- shape: overall 3D form and silhouette. 10 = same form. 5 = clearly the same kind of object but the form is off. 1 = unrelated.
- proportions: relative sizes and positions of the main parts (for example head vs body, limb length and thickness).
- parts: whether the distinctive parts and features of the reference are present and placed correctly. Missing or extra parts lower this.
- color: base colors and where they are (markings, regions, patterns). Judge hue and layout, not brightness from lighting.
- materials: surface appearance, such as fur vs smooth, gloss vs matte, fine texture.

Then give:
- overall: your holistic 1-10 judgment of how well the model matches the reference.
- missing_parts / extra_parts: short noun phrases; empty lists if none.
- top_fixes: up to 3 concrete changes to the Blender model that would most improve the match, most important first.

Be strict. A crude blockout from primitives that has the right parts belongs around 3-5 for shape.
Reserve 9-10 for models whose form is hard to tell apart from the reference."""

PAIRWISE_PROMPT = """You compare two 3D model renders against a reference image of an object.
All images were cropped to the object and put on a neutral grey background.
Decide which candidate reproduces the reference object more faithfully: 3D shape and proportions first,
then parts, then color and materials. Ignore background, lighting and render quality.
Answer "tie" only if you truly cannot tell them apart in quality of match."""


def _criterion_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"reason": {"type": "string"}, "score": {"type": "integer", "description": "1 to 10"}},
        "required": ["reason", "score"],
        "additionalProperties": False,
    }


RUBRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{c: _criterion_schema() for c in CRITERIA},
        "overall": {"type": "integer", "description": "1 to 10"},
        "missing_parts": {"type": "array", "items": {"type": "string"}},
        "extra_parts": {"type": "array", "items": {"type": "string"}},
        "top_fixes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*CRITERIA, "overall", "missing_parts", "extra_parts", "top_fixes"],
    "additionalProperties": False,
}

PAIRWISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "winner": {"type": "string", "enum": ["first", "second", "tie"]}},
    "required": ["reason", "winner"],
    "additionalProperties": False,
}


def make_client() -> Any:
    load_dotenv(find_dotenv(usecwd=True))  # the .env of the project you run from
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise EvalSkipped("OPENROUTER_API_KEY is not set (put it in .env); the judge needs OpenRouter")
    from openai import OpenAI

    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, default_headers={"X-Title": "render-eval"})


def _data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _image_part(img: Image.Image) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": _data_url(img)}}


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)  # tolerate prose or ``` fences around the object
        if not m:
            raise
        return json.loads(m.group(0))


def _chat_json(client: Any, model: str, system: str, content: list[dict[str, Any]], schema: dict[str, Any], name: str):
    """One chat call that returns parsed JSON. Uses strict structured output, falls back to plain JSON mode."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 2000}
    try:
        resp = client.chat.completions.create(
            **kwargs, response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        )
    except Exception as exc:  # some providers reject json_schema; retry asking for JSON in the prompt
        if "response_format" not in str(exc) and "json_schema" not in str(exc):
            raise
        messages[0]["content"] = system + "\n\nReply with only a JSON object matching this schema:\n" + json.dumps(schema)
        resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    usage_d = usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})
    return _parse_json(text), usage_d, getattr(resp, "model", model)


def _clamp_score(x: Any) -> int:
    try:
        return int(min(10, max(1, round(float(x)))))
    except (TypeError, ValueError):
        return 1


def evaluate(pair: ImagePair, cfg: EvalConfig, *, client: Any = None) -> EvalResult:
    client = client or make_client()
    content = [
        {"type": "text", "text": "REFERENCE image (the target object):"},
        _image_part(pair.pil("ref")),
        {"type": "text", "text": "CANDIDATE image (render of the Blender model to grade):"},
        _image_part(pair.pil("ren")),
        {"type": "text", "text": "Grade the candidate against the reference using the rubric."},
    ]
    data, usage, model = _chat_json(client, cfg.judge_model, RUBRIC, content, RUBRIC_SCHEMA, "render_grade")

    scores = {c: _clamp_score((data.get(c) or {}).get("score")) for c in CRITERIA}
    mean = sum(scores.values()) / len(scores)
    metrics: dict[str, Any] = {**scores, "overall": _clamp_score(data.get("overall")), "model": model}
    details = {
        "reasons": {c: (data.get(c) or {}).get("reason", "") for c in CRITERIA},
        "missing_parts": data.get("missing_parts", []),
        "extra_parts": data.get("extra_parts", []),
        "top_fixes": data.get("top_fixes", []),
        "usage": usage,
    }
    return EvalResult(NAME, clip01((mean - 1.0) / 9.0), metrics, details)


def compare(
    reference: ImageInput,
    candidate_a: ImageInput,
    candidate_b: ImageInput,
    cfg: EvalConfig | None = None,
    *,
    client: Any = None,
) -> dict[str, Any]:
    """Pairwise preference between two renders. Returns P(a is closer) in [0, 1] and both verdicts."""
    cfg = cfg or EvalConfig()
    client = client or make_client()
    pa = make_pair(reference, candidate_a, cfg)
    pb = make_pair(reference, candidate_b, cfg)
    ref_img, img_a, img_b = pa.pil("ref"), pa.pil("ren"), pb.pil("ren")

    def ask(first: Image.Image, second: Image.Image) -> tuple[dict[str, Any], dict[str, Any]]:
        content = [
            {"type": "text", "text": "REFERENCE image:"},
            _image_part(ref_img),
            {"type": "text", "text": "FIRST candidate:"},
            _image_part(first),
            {"type": "text", "text": "SECOND candidate:"},
            _image_part(second),
            {"type": "text", "text": "Which candidate matches the reference object better?"},
        ]
        data, usage, _ = _chat_json(client, cfg.judge_model, PAIRWISE_PROMPT, content, PAIRWISE_SCHEMA, "pairwise")
        return data, usage

    ab, usage_ab = ask(img_a, img_b)  # a shown first
    ba, usage_ba = ask(img_b, img_a)  # b shown first

    def points_for_a(winner: str, a_is_first: bool) -> float:
        if winner == "tie":
            return 0.5
        return 1.0 if (winner == "first") == a_is_first else 0.0

    votes = [points_for_a(ab.get("winner", "tie"), True), points_for_a(ba.get("winner", "tie"), False)]
    p_a = sum(votes) / 2.0
    return {
        "p_a_better": p_a,
        "winner": "a" if p_a > 0.5 else "b" if p_a < 0.5 else "tie",
        "consistent": votes[0] == votes[1],
        "verdicts": [
            {"order": "a_first", "winner": ab.get("winner"), "reason": ab.get("reason", "")},
            {"order": "b_first", "winner": ba.get("winner"), "reason": ba.get("reason", "")},
        ],
        "model": cfg.judge_model,
        "usage": [usage_ab, usage_ba],
    }
