"""
MCP server exposing BlenderAgent's tools, for agents that run inside Claude Code (``claude -p``).

Claude Code launches this over stdio (see ``claude_code_loop.py``). Each tool dispatches to the same
``BlenderAgent._call_tool`` the API loops use, so Blender, imagegen and render-eval behave identically;
images (renders, the scored stage view) come back as MCP image content so the model sees them.

    python -m blender_agent.mcp_tools --workspace DIR --reference-json '{...}' --traits-json '{...}' \
        --allowed blender_execute_python,evaluate_render
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Image as MCPImage
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from blender_mcp_connector import BlenderMCPConnector, MCPConfig

from .agent import AgentConfig, AgentTraits, BlenderAgent, HarnessTrace
from .emit import NullEmitter
from .evaluator import EvaluatorClient
from .texture_gen import TextureGenConfig
from .tools import ALWAYS_ALLOWED, TOOL_DEFINITIONS

log = logging.getLogger("refresh.mcp_tools")


def _py_type(spec: dict[str, Any]) -> Any:
    t = spec.get("type")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "object":
        return dict
    if t == "array":
        items = spec.get("items") or {}
        return list[_py_type(items)] if items else list
    return Any


class ToolHost:
    """Holds the agent state shared by every tool call in this Claude Code session."""

    def __init__(self, workspace: str, reference: dict[str, Any], traits: AgentTraits, texture_backend: str | None):
        self.ws = Path(workspace)
        self.ws.mkdir(parents=True, exist_ok=True)
        self.reference = reference
        self.traits = traits
        self.trace = HarnessTrace(goal="", traits_summary=traits.summary())
        self.em = NullEmitter()
        self.iteration = 0
        self.agent = BlenderAgent(
            evaluator=EvaluatorClient.from_render_eval(fast=True),
            config=AgentConfig(llm_backend="claude_code", texture=TextureGenConfig(backend=texture_backend),
                               workspace=str(self.ws)),
        )
        self.agent._blender = BlenderMCPConnector(MCPConfig())
        self.agent._eval_count = 0
        self._lock = threading.Lock()  # one Blender call at a time, even if the client sends tools in parallel

    def call(self, name: str, args: dict[str, Any]) -> list[Any]:
        args = {k: v for k, v in args.items() if v is not None}
        with self._lock:
            log.info("call %s %s", name, json.dumps(args, default=str)[:400])
            try:
                result = self.agent._call_tool(name, args, self.reference, self.ws, self.traits, self.trace,
                                               self.iteration, self.em)
            except Exception as exc:
                log.exception("tool %s failed", name)
                raise ToolError(_reason(exc)) from exc
            finally:
                self.iteration += 1
        images: list[str] = []
        if isinstance(result, dict) and "_images" in result:
            result = dict(result)
            images = [str(p) for p in result.pop("_images") or []]
        out: list[Any] = [result if isinstance(result, str) else json.dumps(result, default=str)]
        out += [MCPImage(path=p) for p in images[:4] if Path(p).is_file()]
        return out


def _reason(exc: Exception) -> str:
    """The error the agent sees: the exception text, with long Blender tracebacks cut to their tail."""
    text = str(exc).strip() or type(exc).__name__
    lines = text.splitlines()
    if len(lines) > 16:
        text = "\n".join(lines[:2] + ["..."] + lines[-12:])
    return f"{type(exc).__name__}: {text}"[:3000]


def _make_tool(host: ToolHost, tdef: dict[str, Any]):
    schema = tdef["input_schema"]
    required = set(schema.get("required", []))
    params = []
    for pname, spec in schema.get("properties", {}).items():
        ann = _py_type(spec)
        if pname in required:
            params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY, annotation=ann))
        else:
            params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY, default=None,
                                            annotation=ann | None))

    def tool(**kwargs: Any) -> list[Any]:
        return host.call(tdef["name"], kwargs)

    tool.__name__ = tdef["name"]
    tool.__doc__ = tdef["description"]
    tool.__signature__ = inspect.Signature(params, return_annotation=list)  # type: ignore[attr-defined]
    tool.__annotations__ = {p.name: p.annotation for p in params} | {"return": list}
    return tool


def build_server(host: ToolHost, allowed: list[str] | None) -> MCPServer:
    server = MCPServer(name="refresh", instructions=(
        "Tools to build, inspect and score a 3-D model in the live Blender scene. The stage camera and "
        "lights (collection RefreshStage) belong to the harness; never modify them."))
    allow = set(allowed or []) | set(ALWAYS_ALLOWED) if allowed else None
    for tdef in TOOL_DEFINITIONS:
        if allow is not None and tdef["name"] not in allow:
            continue
        server.add_tool(_make_tool(host, tdef), name=tdef["name"], description=tdef["description"],
                        structured_output=False)
    return server


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh Blender tools as an MCP server (stdio)")
    p.add_argument("--workspace", required=True)
    p.add_argument("--reference-json", default="{}")
    p.add_argument("--traits-json", default="{}")
    p.add_argument("--allowed", default="", help="comma-separated tool names; empty = all tools")
    p.add_argument("--texture-backend", default=None)
    a = p.parse_args()
    # stderr (tracebacks, library chatter) goes to a log in the workspace; the SDK keeps stdout for JSON-RPC.
    Path(a.workspace).mkdir(parents=True, exist_ok=True)
    log_fd = os.open(str(Path(a.workspace) / "mcp_tools.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    traits_d = json.loads(a.traits_json)
    traits_d["render_angles"] = [tuple(x) for x in traits_d.get("render_angles", [])] or AgentTraits().render_angles
    traits_d["render_resolution"] = tuple(traits_d.get("render_resolution", (512, 512)))
    host = ToolHost(a.workspace, json.loads(a.reference_json), AgentTraits(**traits_d), a.texture_backend or None)
    allowed = [t for t in a.allowed.split(",") if t] or None
    build_server(host, allowed).run("stdio")


if __name__ == "__main__":
    main()
