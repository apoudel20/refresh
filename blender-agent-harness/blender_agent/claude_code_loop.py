"""
Agent loop on the Claude Code CLI (``claude -p``): agents run on the user's Claude subscription, no API key.

Claude Code drives the tool loop itself. Our tools reach it through the stdio MCP server in ``mcp_tools.py``,
which runs them with the same ``BlenderAgent._call_tool`` dispatch the API loops use. This module launches the
CLI, turns its stream-json output into Emitter events and a HarnessTrace, and ends the session early on the
traits' target score or plateau, like the other loops.

Auth: whatever the CLI is logged in with (`claude` then /login, or `claude auth login`), or a long-lived
`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`. ANTHROPIC_API_KEY is removed from the CLI's environment so
it never silently bills an API key instead of the subscription.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .evaluator import EvaluationResult
from .tools import ALWAYS_ALLOWED, TOOL_DEFINITIONS

if TYPE_CHECKING:
    from .agent import AgentTraits, BlenderAgent, HarnessTrace
    from .emit import Emitter

SERVER = "refresh"
PREFIX = f"mcp__{SERVER}__"
# Removed from the CLI's environment: API credentials (they'd take precedence over the subscription) and the
# markers of an enclosing Claude Code session (nested sessions refuse to start).
STRIP_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
AUTH_MARKERS = ("not logged in", "/login", "invalid api key", "oauth token", "authentication_error",
                "authentication failed", "credit balance")
BUILTIN_TOOLS = ("Read", "Glob")  # look at images, list files; everything else goes through the refresh tools
LIMIT_MARKERS = ("usage limit", "limit reached", "limit will reset", "out of extra usage")


def claude_bin() -> str | None:
    """The Claude Code executable: $CLAUDE_BIN, PATH, or the default install location."""
    for cand in (os.getenv("CLAUDE_BIN"), shutil.which("claude"), str(Path.home() / ".local/bin/claude")):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV and (v or not k.startswith("CLAUDE_"))}
    env.setdefault("MCP_TIMEOUT", "120000")          # tool server start-up (imports render-eval, imagegen)
    env.setdefault("MCP_TOOL_TIMEOUT", "1800000")    # renders, turntables and texture generation are slow
    env.setdefault("MAX_MCP_OUTPUT_TOKENS", "60000")  # renders come back as images
    env.update(extra or {})
    return env


def auth_status(timeout: float = 20.0) -> dict[str, Any]:
    """`claude auth status` as a dict: {"loggedIn": bool, "authMethod": ...} (plus "error" when unknown)."""
    exe = claude_bin()
    if not exe:
        return {"loggedIn": False, "error": "Claude Code CLI not found (install it or set CLAUDE_BIN)"}
    if os.getenv("CLAUDE_CODE_OAUTH_TOKEN"):
        return {"loggedIn": True, "authMethod": "oauth_token (CLAUDE_CODE_OAUTH_TOKEN)", "bin": exe}
    try:
        out = subprocess.run([exe, "auth", "status"], capture_output=True, text=True, timeout=timeout,
                             env=child_env(), stdin=subprocess.DEVNULL)
        data = json.loads(out.stdout or "{}")
        return {**data, "bin": exe} if isinstance(data, dict) else {"loggedIn": False, "error": out.stdout[:300]}
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        return {"loggedIn": False, "error": f"{type(exc).__name__}: {exc}", "bin": exe}


def model_name(model: str | None) -> str | None:
    """CLI model argument from a configured name: drops a provider prefix ("anthropic/claude-opus-5")."""
    model = (model or "").strip()
    if not model or model in ("default", "cli"):
        return None
    model = model.split("/", 1)[1] if model.startswith("anthropic/") else model
    return re.sub(r"(\d)\.(\d)", r"\1-\2", model) if model.startswith("claude-") else model


def is_auth_failure(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in AUTH_MARKERS)


def is_limit_failure(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in LIMIT_MARKERS)


# ── the agent loop ────────────────────────────────────────────────────────


def run(agent: "BlenderAgent", goal: str, reference: dict[str, Any], traits: "AgentTraits", ws: Path,
        trace: "HarnessTrace", em: "Emitter") -> None:
    from .agent import ScoreEvent, ToolEvent

    exe = claude_bin()
    if not exe:
        raise ConnectionError("Claude Code CLI not found: install Claude Code or set CLAUDE_BIN to its path")

    names = {t["name"] for t in TOOL_DEFINITIONS}
    allowed = sorted((set(traits.allowed_tools) & names if traits.allowed_tools else names) | set(ALWAYS_ALLOWED))
    mcp_cfg = ws / "mcp.json"
    mcp_cfg.write_text(json.dumps({"mcpServers": {SERVER: _server_entry(agent, reference, traits, ws, allowed)}},
                                  indent=2))

    max_turns = int(os.getenv("REFRESH_AGENT_MAX_TURNS", "0"))  # 0 = no turn limit
    cmd = [exe, "-p",
           "--dangerously-skip-permissions",
           "--output-format", "stream-json", "--verbose",
           "--mcp-config", str(mcp_cfg),
           "--append-system-prompt", traits.build_system_prompt() + CLAUDE_CODE_NOTES,
           "--no-session-persistence"]
    for d in _read_dirs(reference, ws):
        cmd += ["--add-dir", d]
    if max_turns > 0:
        cmd += ["--max-turns", str(max_turns)]
    model = model_name(agent.cfg.model)
    if model:
        cmd += ["--model", model]
    if agent.cfg.effort:
        cmd += ["--effort", agent.cfg.effort]
    (ws / "claude_cmd.json").write_text(json.dumps(cmd[1:], indent=2))

    proc = subprocess.Popen(cmd, cwd=str(ws), env=child_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
    assert proc.stdin and proc.stdout and proc.stderr
    proc.stdin.write(_prompt(goal, reference, traits, ws))
    proc.stdin.close()

    stderr_tail: list[str] = []
    threading.Thread(target=_drain, args=(proc.stderr, stderr_tail), daemon=True).start()
    timeout = float(os.getenv("REFRESH_AGENT_TIMEOUT", "10800"))  # wall-clock safety net for a hung session
    timed_out = threading.Event()

    def _on_timeout() -> None:
        timed_out.set()
        _kill(proc)

    watchdog = threading.Timer(timeout, _on_timeout)
    watchdog.daemon = True
    watchdog.start()

    pending: dict[str, tuple[str, dict[str, Any], float, int]] = {}
    recent: list[float] = []
    turn = 0
    final: dict[str, Any] | None = None
    init: dict[str, Any] = {}
    last_text = ""
    stop_reason = ""
    log = (ws / "claude_stream.jsonl").open("w")
    try:
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                log.write(line[:2000])
                continue
            log.write(json.dumps(_strip_images(ev)) + "\n")
            kind = ev.get("type")
            if kind == "system" and ev.get("subtype") == "init":
                init = ev
                failed = [s for s in ev.get("mcp_servers", [])
                          if s.get("name") == SERVER and s.get("status") in ("failed", "needs-auth")]
                if failed:
                    _kill(proc)
                    raise RuntimeError(f"the refresh tool server did not start ({failed[0].get('status')}); "
                                       f"see {ws / 'mcp_tools.log'}")
            elif kind == "assistant" and not ev.get("parent_tool_use_id"):
                msg = ev.get("message") or {}
                uses = [b for b in msg.get("content", []) if b.get("type") == "tool_use"]
                # what the agent says and thinks, in order, for the dashboard's transcript
                for b in msg.get("content", []):
                    if b.get("type") == "thinking" and (b.get("thinking") or "").strip():
                        em.emit("thinking", iteration=turn, text=b["thinking"].strip()[:6000])
                    elif b.get("type") == "text" and (b.get("text") or "").strip():
                        em.emit("message", iteration=turn, text=b["text"].strip()[:6000])
                if uses:
                    turn += 1
                    trace.total_iterations = turn
                for b in uses:
                    tool = _short_name(b.get("name", ""))
                    args = b.get("input") or {}
                    pending[b.get("id", "")] = (tool, args, time.perf_counter(), turn - 1)
                    em.tool_call(turn - 1, tool, args)
                text = " ".join(b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text").strip()
                if text:
                    last_text = text
                if text and (is_auth_failure(text) or is_limit_failure(text)) and not uses:
                    stderr_tail.append(text)
            elif kind == "user" and not ev.get("parent_tool_use_id"):
                for b in (ev.get("message") or {}).get("content", []) or []:
                    if not isinstance(b, dict) or b.get("type") != "tool_result":
                        continue
                    tool, args, t0, it = pending.pop(b.get("tool_use_id", ""), ("?", {}, time.perf_counter(), turn))
                    text = _result_text(b.get("content"))
                    error = text[:500] if b.get("is_error") else None
                    duration_ms = (time.perf_counter() - t0) * 1000
                    em.tool_result(it, tool, duration_ms, error, text[:1500] + ("…" if len(text) > 1500 else ""))
                    result = _parse(text)
                    trace.tool_events.append(ToolEvent(iteration=it, tool=tool, args=args, result=result,
                                                       error=error, duration_ms=duration_ms))
                    if not error:
                        _record_artifacts(tool, result, em)
                    if tool == "evaluate_render" and not error and isinstance(result, dict):
                        score = _record_score(result, it, trace, em, ScoreEvent)
                        recent.append(score)
                        stop_reason = _stop_check(recent, traits) if _early_stop() else ""
                        if stop_reason:
                            _kill(proc)
            elif kind == "result":
                final = ev
    finally:
        watchdog.cancel()
        log.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _kill(proc, signal.SIGKILL)
            proc.wait()

    if final:
        trace.cost_usd += float(final.get("total_cost_usd") or 0.0)
        usage = final.get("usage") or {}
        trace.input_tokens += int(usage.get("input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0) \
            + int(usage.get("cache_creation_input_tokens") or 0)
        trace.output_tokens += int(usage.get("output_tokens") or 0)
        trace.total_iterations = int(final.get("num_turns") or trace.total_iterations)

    trace.final_text = (str(final.get("result") or "") if final and not final.get("is_error") else "") or last_text
    if stop_reason:
        trace.stop_reason = stop_reason
        return
    if timed_out.is_set():
        trace.stop_reason = f"timeout ({int(timeout)}s)"
        return

    failed = final is None or bool(final.get("is_error"))
    text = (str((final or {}).get("result") or "") + " " + " ".join(stderr_tail[-20:])).strip()
    if failed and is_auth_failure(text):
        raise ConnectionError("Claude Code is not logged in for the process running refresh-server "
                              f"({text[:200]}). Run `claude` then /login (or `claude auth login`) in that "
                              "terminal, or set CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`.")
    if failed and is_limit_failure(text):
        raise ConnectionError(f"Claude subscription usage limit reached: {text[:300]}")
    if final is None:
        raise RuntimeError(f"claude -p exited ({proc.returncode}) without a result: {text[:600]}")
    sub = final.get("subtype") or ""
    if sub == "error_max_turns":
        trace.stop_reason = "max_iterations"
        return
    if failed or sub != "success":
        raise RuntimeError(f"claude -p failed ({sub or 'error'}, model {init.get('model', '?')}): {text[:600]}")
    trace.stop_reason = "model_stop"


CLAUDE_CODE_NOTES = (
    "\n\nYou are running headless inside the Refresh harness: there is no user to ask, so never wait for input. "
    "You have Claude Code's tools (shell, files, web, skills), the `refresh` MCP tools (Blender through the "
    "harness, imagegen, `evaluate_render` scoring) and any other MCP servers configured on this machine. Keep all "
    "files you write inside your folder. Build the model in the live Blender scene: the harness snapshots that "
    "scene when you finish, so work saved only to files is not handed on. Never quit Blender, open another file "
    "in it, or reset its scene. When you finish, reply with a short summary of what you built and what still "
    "differs from the reference."
)


def _prompt(goal: str, reference: dict[str, Any], traits: "AgentTraits", ws: Path) -> str:
    lines = [f"Goal: {goal}",
             f"Your folder: {ws}",
             "Work only inside your folder: write every script to scripts/, debug output, logs and scratch files to "
             "debug/, your own renders and images to renders/, and .blend checkpoints to blend/ (save them with "
             "bpy.ops.wm.save_as_mainfile(filepath=..., copy=True); copy=True matters). Do not create, change or "
             "delete anything outside your folder, except reading the reference photo and other files you need.",
             "The harness saved the scene you were handed as blend/start.blend and saves blend/end.blend when you "
             "finish. The team's work so far is already loaded in the live Blender scene; there are no model files "
             "to import."]
    if reference.get("image_path"):
        lines.append(f"Reference photo (the target): {reference['image_path']} -- look at it first with Read.")
    views = [v for v in reference.get("extra_views", []) if v]
    if views:
        lines.append("Extra reference views (context only; scoring uses the main photo): " + ", ".join(views))
    if reference.get("brief_extra"):
        lines.append(reference["brief_extra"])
    lines.append("There is no turn limit: take as long as the model needs. Inspect the scene, make your changes, "
                 "call evaluate_render after each significant change, and keep improving until you are satisfied "
                 "it looks like the object from every side. Begin.")
    return "\n".join(lines)


def _server_entry(agent: "BlenderAgent", reference: dict[str, Any], traits: "AgentTraits", ws: Path,
                  allowed: list[str]) -> dict[str, Any]:
    traits_d = dataclasses.asdict(traits)
    args = ["-m", "blender_agent.mcp_tools", "--workspace", str(ws),
            "--reference-json", json.dumps(reference, default=str), "--traits-json", json.dumps(traits_d),
            "--allowed", ",".join(allowed)]
    if agent.cfg.texture.backend:
        args += ["--texture-backend", agent.cfg.texture.backend]
    mcp = agent.cfg.mcp
    env = {"BLENDER_MCP_HOST": str(mcp.host), "BLENDER_MCP_PORT": str(mcp.port),
           "BLENDER_MCP_PROTOCOL": str(mcp.protocol), "PYTHONUNBUFFERED": "1"}
    return {"type": "stdio", "command": sys.executable, "args": args, "env": env}


def _read_dirs(reference: dict[str, Any], ws: Path) -> list[str]:
    dirs = {str(ws)}
    for p in [reference.get("image_path"), *(reference.get("extra_views") or [])]:
        if p:
            dirs.add(str(Path(p).resolve().parent))
    return sorted(dirs)


def _record_score(result: dict[str, Any], it: int, trace: "HarnessTrace", em: "Emitter", ScoreEvent: Any) -> float:
    sub = result.get("subscores") or {}
    overall = float(result.get("overall_score") or 0.0)
    feedback = [str(f) for f in result.get("priority_feedback") or []]
    ev = EvaluationResult(overall_score=overall, visual_fidelity=float(sub.get("visual", 0.0)),
                          topology_quality=float(sub.get("topology", 0.0)),
                          depth_alignment=float(sub.get("depth", 0.0)),
                          vertex_accuracy=float(sub.get("vertices", 0.0)), feedback=feedback,
                          scores=dict(result.get("step_scores") or {}))
    trace.final_evaluation = ev
    trace.score_events.append(ScoreEvent(iteration=it, overall=overall, visual=ev.visual_fidelity,
                                         topology=ev.topology_quality, depth=ev.depth_alignment,
                                         vertices=ev.vertex_accuracy, feedback=feedback))
    em.score(iteration=it, overall=overall, visual=ev.visual_fidelity, topology=ev.topology_quality,
             depth=ev.depth_alignment, vertices=ev.vertex_accuracy, feedback=feedback)
    return overall


def _record_artifacts(tool: str, result: Any, em: "Emitter") -> None:
    """Images a tool produced, for the dashboard's live gallery (the tool server itself can't log events)."""
    if not isinstance(result, dict):
        return
    if tool == "evaluate_render":
        meta = {"overall": result.get("overall_score"), "front_match": result.get("front_match"),
                "solidity": (result.get("solidity") or {}).get("solidity")}
        if result.get("stage_render"):
            em.artifact("stage_render", result["stage_render"], meta)
        if result.get("turntable"):
            em.artifact("turntable", result["turntable"], meta)
    elif tool == "blender_render":
        for p in result.get("paths") or []:
            em.artifact("render", p)
    elif result.get("path") or result.get("atlas"):
        em.artifact("image", result.get("path") or result.get("atlas"), {"tool": tool})


def _early_stop() -> bool:
    """REFRESH_AGENT_EARLY_STOP=on ends a session at the target score or on a score plateau (default: off)."""
    return os.getenv("REFRESH_AGENT_EARLY_STOP", "off").strip().lower() in ("1", "on", "true", "yes")


def _stop_check(recent: list[float], traits: "AgentTraits") -> str:
    latest = recent[-1]
    if latest >= traits.target_score:
        return f"target_score_reached ({latest:.3f})"
    if len(recent) >= max(2, traits.plateau_window):
        window = recent[-traits.plateau_window:]
        if max(window) - min(window) < traits.plateau_min_delta:
            return f"plateau ({[round(w, 3) for w in window]})"
    return ""


def _short_name(name: str) -> str:
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return "" if content is None else str(content)


def _parse(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _strip_images(x: Any) -> Any:
    """The stream echoes images as base64; keep the log readable."""
    if isinstance(x, dict):
        if isinstance(x.get("data"), str) and (x.get("type") in ("base64", "image") or "mimeType" in x):
            return {**x, "data": f"<{len(x['data'])} chars>"}
        return {k: _strip_images(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_images(v) for v in x]
    return x


def _drain(stream: Any, tail: list[str]) -> None:
    for line in stream:
        tail.append(line.rstrip())
        del tail[:-50]


def _kill(proc: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    """Stop the CLI and its tool server (its own process group)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass
