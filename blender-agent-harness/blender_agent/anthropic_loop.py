"""
BlenderAgent's tool loop on the Anthropic Messages API.

A manual loop (not the SDK tool runner) because each turn has harness work around it:
score-based stopping (target / plateau), event emission to the dashboard, and tool results
that carry images (renders) so the model can see its own work.

Per current API guidance:
- streamed requests with ``eager_input_streaming`` on every tool, so each tool input is
  validated against its schema here before anything runs;
- adaptive thinking + ``output_config.effort``;
- server-side refusal fallbacks (``fallbacks: "default"``, beta
  ``server-side-fallback-2026-07-01``), with fallback turns echoed back per the rules
  (thinking/tool_use blocks before the last ``fallback`` block are dropped);
- automatic prompt caching (stable system prompt + tool list).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tools import anthropic_tools

if TYPE_CHECKING:
    from .agent import AgentTraits, BlenderAgent, HarnessTrace
    from .emit import Emitter

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# $ per million tokens (input, output) for cost tracking; cache reads bill ~0.1x input, writes ~1.25x.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

MAX_IMAGE_BYTES = 4_500_000
MAX_TEXT_CHARS = 20_000


def usage_cost(model: str, usage: Any) -> float:
    pin, pout = PRICES.get(model, (5.0, 25.0))
    g = lambda k: float(getattr(usage, k, 0) or 0)  # noqa: E731
    return (g("input_tokens") * pin + g("output_tokens") * pout
            + g("cache_read_input_tokens") * pin * 0.1 + g("cache_creation_input_tokens") * pin * 1.25) / 1e6


def image_block(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file() or p.stat().st_size > MAX_IMAGE_BYTES:
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        return None
    data = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}


def validate_input(schema: dict[str, Any], data: Any) -> str | None:
    """Minimal JSON-schema check for tool inputs (object, required keys, primitive types, enums)."""
    if not isinstance(data, dict):
        return "input must be a JSON object"
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in data:
            return f"missing required field {key!r}"
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
    for key, value in data.items():
        spec = props.get(key)
        if not spec:
            continue
        want = types.get(spec.get("type", ""))
        if want and not isinstance(value, want) or (spec.get("type") in ("integer", "number") and isinstance(value, bool)):
            return f"field {key!r} should be {spec.get('type')}"
        if "enum" in spec and value not in spec["enum"]:
            return f"field {key!r} must be one of {spec['enum']}"
    return None


def echo_content(content: list[Any]) -> list[Any]:
    """Assistant content to send back. After a mid-output fallback, drop thinking/tool_use blocks
    that precede the last ``fallback`` marker (the fallback model restarted from there)."""
    idx = max((i for i, b in enumerate(content) if getattr(b, "type", "") == "fallback"), default=-1)
    if idx < 0:
        return list(content)
    keep_before = {"text"}
    return [b for i, b in enumerate(content) if i > idx or getattr(b, "type", "") in keep_before]


def executable_tool_uses(content: list[Any]) -> list[Any]:
    idx = max((i for i, b in enumerate(content) if getattr(b, "type", "") == "fallback"), default=-1)
    return [b for i, b in enumerate(content) if i > idx and getattr(b, "type", "") == "tool_use"]


def to_tool_content(result: Any) -> list[dict[str, Any]] | str:
    """Tool result -> tool_result content. Dicts may carry '_images': [paths] to show the model."""
    images: list[str] = []
    if isinstance(result, dict) and "_images" in result:
        result = dict(result)
        images = [str(p) for p in result.pop("_images") or []]
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "... [truncated]"
    blocks = [b for b in (image_block(p) for p in images[:4]) if b]
    if not blocks:
        return text or "(no output)"
    return [{"type": "text", "text": text or "(no output)"}, *blocks]


def initial_content(goal: str, reference: dict[str, Any], ws: Path, extra: str = "") -> list[dict[str, Any]]:
    lines = [f"Goal: {goal}", f"Workspace directory (write files here): {ws}"]
    if reference.get("image_path"):
        lines.append(f"Reference image (the target, shown below): {reference['image_path']}")
    views = [v for v in reference.get("extra_views", []) if v]
    if views:
        lines.append("Extra reference views (context only; scoring uses the main reference): " + ", ".join(views))
    if extra:
        lines.append(extra)
    lines.append("Begin.")
    blocks: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    for p in [reference.get("image_path"), *views[:3]]:
        b = image_block(p) if p else None
        if b:
            blocks.append(b)
    return blocks


def _request(agent: "BlenderAgent", system: list[dict[str, Any]], tools: list[dict[str, Any]],
             messages: list[dict[str, Any]]) -> Any:
    cfg = agent.cfg
    kwargs: dict[str, Any] = dict(model=cfg.model, max_tokens=cfg.max_tokens, system=system,
                                  tools=tools, messages=messages)
    extra: dict[str, Any] = {"cache_control": {"type": "ephemeral"}}
    if not cfg.model.startswith("claude-haiku"):
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": cfg.effort}
    betas: list[str] = []
    if cfg.anthropic_fallbacks:
        betas.append(FALLBACK_BETA)
        extra["fallbacks"] = "default"
    if betas:
        kwargs["betas"] = betas
    kwargs["extra_body"] = extra

    for attempt in range(3):
        try:
            with agent._anthropic.beta.messages.stream(**kwargs) as stream:
                return stream.get_final_message()
        except ValueError:
            # A tool input the SDK could not parse at all (eager input streaming); no tool_use id to
            # answer, so re-issue the request. Typed API errors (rate limit, auth) propagate.
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")


def run(agent: "BlenderAgent", goal: str, reference: dict[str, Any], traits: "AgentTraits", ws: Path,
        trace: "HarnessTrace", em: "Emitter") -> None:
    from .agent import ToolEvent

    tools = anthropic_tools(traits.allowed_tools, eager_input_streaming=True)
    schemas = {t["name"]: t["input_schema"] for t in tools}
    system = [{"type": "text", "text": traits.build_system_prompt()}]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": initial_content(goal, reference, ws, reference.get("brief_extra", ""))}
    ]
    recent: list[float] = []

    for iteration in range(traits.max_iterations):
        trace.total_iterations = iteration + 1
        response = _request(agent, system, tools, messages)
        trace.cost_usd += usage_cost(response.model or agent.cfg.model, response.usage)
        trace.input_tokens += int(getattr(response.usage, "input_tokens", 0) or 0)
        trace.output_tokens += int(getattr(response.usage, "output_tokens", 0) or 0)

        if response.stop_reason == "refusal":
            trace.stop_reason = f"refusal ({getattr(response.stop_details, 'category', None)})"
            break

        messages.append({"role": "assistant", "content": echo_content(response.content)})
        uses = executable_tool_uses(response.content)
        for b in response.content:  # what the agent says and thinks, for the dashboard's transcript
            if b.type == "thinking" and (getattr(b, "thinking", "") or "").strip():
                em.emit("thinking", iteration=iteration, text=b.thinking.strip()[:6000])
            elif b.type == "text" and (b.text or "").strip():
                em.emit("message", iteration=iteration, text=b.text.strip()[:6000])

        if response.stop_reason == "max_tokens" and uses:
            # A truncated tool input can parse as a valid-looking partial object: never run it.
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": u.id, "is_error": True,
                 "content": "Your output hit max_tokens inside this tool call. Send a shorter call."}
                for u in uses]})
            continue
        if not uses:
            trace.stop_reason = "model_stop"
            break

        results: list[dict[str, Any]] = []
        for u in uses:
            args = u.input if isinstance(u.input, dict) else {}
            em.tool_call(iteration, u.name, args)
            t0 = time.perf_counter()
            error: str | None = None
            result: Any = None
            invalid = validate_input(schemas[u.name], u.input) if u.name in schemas else f"unknown tool {u.name!r}"
            if invalid:
                error = invalid
                content: Any = json.dumps({"INVALID_JSON": json.dumps(u.input, default=str)[:4000], "error": invalid})
            else:
                try:
                    result = agent._call_tool(u.name, args, reference, ws, traits, trace, iteration, em)
                    content = to_tool_content(result)
                except Exception as exc:  # the model reads the error and tries something else
                    error = str(exc)
                    content = f"Error: {exc}"
            ms = (time.perf_counter() - t0) * 1000
            summary = content if isinstance(content, str) else content[0]["text"]
            em.tool_result(iteration, u.name, ms, error, summary[:1500] + ("…" if len(summary) > 1500 else ""))
            trace.tool_events.append(ToolEvent(iteration=iteration, tool=u.name, args=args, result=result,
                                               error=error, duration_ms=ms))
            block: dict[str, Any] = {"type": "tool_result", "tool_use_id": u.id, "content": content}
            if error:
                block["is_error"] = True
            results.append(block)
        messages.append({"role": "user", "content": results})  # all results in one message

        if trace.score_events:
            latest = trace.score_events[-1].overall
            recent.append(latest)
            if latest >= traits.target_score:
                trace.stop_reason = f"target_score_reached ({latest:.3f})"
                break
            if len(recent) >= traits.plateau_window:
                window = recent[-traits.plateau_window:]
                if max(window) - min(window) < traits.plateau_min_delta:
                    trace.stop_reason = f"plateau ({[round(x, 3) for x in window]})"
                    break
    else:
        trace.stop_reason = "max_iterations"
