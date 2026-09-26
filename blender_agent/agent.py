"""
BlenderAgent — single agentic loop driven by AgentTraits.

No orchestrator, no sub-agents.  Traits shape the agent's behaviour (system
prompt injection, tool allow-list, stopping policy, scoring weights).  The
harness records every tool call and score into a HarnessTrace so callers can
inspect exactly what the agent did and why.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from . import secrets as _secrets
from .emit import Emitter, NullEmitter
from .evaluator import EvaluatorClient, EvaluationResult, RenderPayload
from .feedback import FeedbackAccessor, FeedbackCategory
from .mcp_connector import BlenderMCPConnector, MCPConfig
from .pointcloud import ImageToPointCloud, PointCloudConfig
from .texture_gen import TextureGenerator, TextureGenConfig
from .tools import openai_tools

LLMBackend = Literal["openai", "openrouter"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ── Harness trace ──────────────────────────────────────────────────────────

@dataclass
class ToolEvent:
    """One tool call emitted by the agent."""
    iteration: int
    tool: str
    args: dict[str, Any]
    result: Any
    error: str | None
    duration_ms: float


@dataclass
class ScoreEvent:
    """One evaluation result received by the agent."""
    iteration: int
    overall: float
    visual: float
    topology: float
    depth: float
    vertices: float
    feedback: list[str]


@dataclass
class HarnessTrace:
    """Complete record of one agent run — what it did and how it performed."""
    goal: str
    traits_summary: str
    tool_events: list[ToolEvent] = field(default_factory=list)
    score_events: list[ScoreEvent] = field(default_factory=list)
    final_evaluation: EvaluationResult | None = None
    total_iterations: int = 0
    stop_reason: str = ""

    # ── Derived views ──────────────────────────────────────────────

    def tool_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.tool_events:
            counts[e.tool] = counts.get(e.tool, 0) + 1
        return counts

    def score_progression(self) -> list[float]:
        return [e.overall for e in self.score_events]

    def errors(self) -> list[ToolEvent]:
        return [e for e in self.tool_events if e.error]

    def slowest_tools(self, n: int = 5) -> list[ToolEvent]:
        return sorted(self.tool_events, key=lambda e: e.duration_ms, reverse=True)[:n]


# ── Agent traits ───────────────────────────────────────────────────────────

@dataclass
class AgentTraits:
    """
    Behavioural parameters injected into the agent.  Change these to change
    what the agent prioritises, how aggressive it is, and when it stops.

    All fields have sensible defaults — only override what you need.
    """

    # ── Priority weights (sum doesn't need to be 1) ────────────────
    # Higher = agent will push harder on that dimension before moving on.
    weight_topology: float  = 1.0
    weight_texture: float   = 1.0
    weight_shape: float     = 1.0
    weight_depth: float     = 0.5

    # ── Stopping policy ────────────────────────────────────────────
    target_score: float     = 0.80   # stop when overall score >= this
    max_iterations: int     = 8      # hard cap regardless of score
    # Stop if score hasn't improved by at least this much in the last N rounds
    plateau_min_delta: float = 0.02
    plateau_window: int      = 3

    # ── Tool allow-list (None = all tools allowed) ─────────────────
    # Restrict to a subset of tool names to limit what the agent can do.
    allowed_tools: list[str] | None = None

    # ── Free-text persona injected into system prompt ──────────────
    # Examples: "prefer quad topology", "keep vertex count under 5000",
    #           "use CYCLES renderer only", "match the silhouette first"
    persona: str = ""

    # ── Render config passed to blender_render tool ────────────────
    render_angles: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0, 0, 0), (0, 45, 0), (0, 90, 0), (0, 135, 0)]
    )
    render_resolution: tuple[int, int] = (512, 512)

    def summary(self) -> str:
        parts = [
            f"weights(topo={self.weight_topology} tex={self.weight_texture} "
            f"shape={self.weight_shape} depth={self.weight_depth})",
            f"target={self.target_score} max_iter={self.max_iterations}",
        ]
        if self.persona:
            parts.append(f'persona="{self.persona}"')
        if self.allowed_tools:
            parts.append(f"tools={self.allowed_tools}")
        return "  ".join(parts)

    def active_tools(self) -> list[dict[str, Any]]:
        """Return the OpenAI tool list filtered by allowed_tools."""
        all_tools = openai_tools()
        if not self.allowed_tools:
            return all_tools
        allowed = set(self.allowed_tools)
        return [t for t in all_tools if t["function"]["name"] in allowed]

    def build_system_prompt(self) -> str:
        base = (
            "You are an expert 3-D modelling assistant operating inside a Blender "
            "automation pipeline.  Your job is to recreate a 3-D model that matches "
            "a reference description or image as closely as possible.\n\n"
            "Workflow:\n"
            "1. Use `image_to_pointcloud` (if a reference image is given) to bootstrap geometry.\n"
            "2. Import the PLY with `blender_import_file`.\n"
            "3. Clean up topology with `blender_smooth_mesh` / `blender_apply_subdivision`.\n"
            "4. Unwrap UVs with `blender_unwrap_uv`.\n"
            "5. Generate textures with `generate_texture` (albedo first, then normal/roughness).\n"
            "6. Assign with `blender_set_material`.\n"
            "7. Render from multiple angles with `blender_render`.\n"
            "8. Call `evaluate_render` and read the scores and feedback carefully.\n"
            "9. Act on the highest-priority feedback and repeat from step 3.\n\n"
        )

        # Inject priority guidance from weights
        priorities: list[str] = []
        ranked = sorted(
            [
                ("topology / mesh quality", self.weight_topology),
                ("texture and material quality", self.weight_texture),
                ("overall shape and silhouette", self.weight_shape),
                ("depth and relief accuracy", self.weight_depth),
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        for name, w in ranked:
            if w >= 0.8:
                priorities.append(f"PRIORITISE {name} (weight {w:.1f})")
            elif w <= 0.3:
                priorities.append(f"de-prioritise {name} (weight {w:.1f})")

        if priorities:
            base += "Priority guidance (from traits):\n"
            base += "\n".join(f"  • {p}" for p in priorities) + "\n\n"

        if self.persona:
            base += f"Additional constraints:\n  {self.persona}\n\n"

        base += (
            "Always call `evaluate_render` after each render batch.\n"
            "On tool errors, diagnose and try an alternative — never repeat the same failing call.\n"
            "One logical action per tool call.\n"
            "NEVER call bpy.ops.wm.read_factory_settings, bpy.ops.wm.read_homefile, "
            "or any other scene-reset operation — it kills the MCP server connection."
        )
        return base


# ── Agent config ───────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    llm_backend: LLMBackend = "openrouter"
    max_tokens: int = 4096
    mcp: MCPConfig = field(default_factory=MCPConfig)
    texture: TextureGenConfig = field(default_factory=TextureGenConfig)
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    workspace: str = "/tmp/blender_agent"


# ── Agent ──────────────────────────────────────────────────────────────────

class BlenderAgent:
    """
    Single agentic loop.  Behaviour is shaped entirely by AgentTraits.
    Returns (EvaluationResult | None, HarnessTrace) so callers get both
    the outcome and a full record of what happened.

    Usage:
        agent = BlenderAgent(evaluator=EvaluatorClient.from_openrouter())
        traits = AgentTraits(
            weight_topology=1.5,
            persona="prefer quad topology, keep faces under 10k",
            target_score=0.82,
        )
        result, trace = agent.run(
            goal="Recreate Suzanne the monkey head",
            reference={"image_path": "front.png"},
            traits=traits,
        )
        print(trace.score_progression())
        print(trace.tool_counts())
    """

    def __init__(
        self,
        evaluator: EvaluatorClient,
        config: AgentConfig | None = None,
        api_key: str = "",
    ):
        self.evaluator = evaluator
        self.cfg = config or AgentConfig()
        _secrets.load()
        self._llm = self._build_client(api_key)
        self._texture_gen = TextureGenerator(self.cfg.texture)
        self._pointcloud  = ImageToPointCloud(self.cfg.pointcloud)

    # ── Public API ─────────────────────────────────────────────────

    def run(
        self,
        goal: str,
        reference: dict[str, Any] | None = None,
        traits: AgentTraits | None = None,
        workspace: str | None = None,
        emitter: Emitter | None = None,
    ) -> tuple[EvaluationResult | None, HarnessTrace]:
        traits  = traits or AgentTraits()
        ws      = Path(workspace or self.cfg.workspace)
        ws.mkdir(parents=True, exist_ok=True)
        em      = emitter or NullEmitter()
        trace   = HarnessTrace(goal=goal, traits_summary=traits.summary())

        em.run_start(goal, traits.summary(), self.cfg.model, self.cfg.llm_backend)

        self._preflight(self.cfg.mcp)

        with BlenderMCPConnector(self.cfg.mcp) as blender:
            self._blender = blender
            self._loop(goal, reference or {}, traits, ws, trace, em)
            self._blender = None

        em.stop(trace.stop_reason, trace.total_iterations,
                trace.final_evaluation.overall_score if trace.final_evaluation else None)
        em.run_end({
            "tool_counts":        trace.tool_counts(),
            "score_progression":  trace.score_progression(),
            "errors":             [{"tool": e.tool, "error": e.error} for e in trace.errors()],
            "stop_reason":        trace.stop_reason,
        })

        return trace.final_evaluation, trace

    # ── Agentic loop ───────────────────────────────────────────────

    def _loop(
        self,
        goal: str,
        reference: dict[str, Any],
        traits: AgentTraits,
        ws: Path,
        trace: HarnessTrace,
        em: Emitter,
    ) -> None:
        history: list[dict[str, Any]] = [
            {"role": "user", "content": self._initial_message(goal, reference, ws)}
        ]
        tools = traits.active_tools()
        recent_scores: list[float] = []

        for iteration in range(traits.max_iterations):
            trace.total_iterations = iteration + 1

            response = self._llm.chat.completions.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                tools=tools,
                tool_choice="auto",
                messages=[{"role": "system", "content": traits.build_system_prompt()}]
                + history,
            )

            msg    = response.choices[0].message
            finish = response.choices[0].finish_reason

            assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            history.append(assistant_entry)

            if finish == "stop" or not msg.tool_calls:
                trace.stop_reason = "model_stop"
                break

            # Dispatch tool calls, emit events, record into harness
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                t0   = time.perf_counter()
                error: str | None = None
                result: Any = None

                em.tool_call(iteration, tc.function.name, args)

                try:
                    result = self._call_tool(tc.function.name, args, reference, ws, traits, trace, iteration, em)
                    content = json.dumps(result, default=str)
                    result_summary = content[:120] + ("…" if len(content) > 120 else "")
                except Exception as exc:
                    error          = str(exc)
                    content        = f"Error: {exc}"
                    result_summary = error

                duration_ms = (time.perf_counter() - t0) * 1000
                em.tool_result(iteration, tc.function.name, duration_ms, error, result_summary)

                trace.tool_events.append(ToolEvent(
                    iteration=iteration,
                    tool=tc.function.name,
                    args=args,
                    result=result,
                    error=error,
                    duration_ms=duration_ms,
                ))

                history.append({"role": "tool", "tool_call_id": tc.id, "content": content})

            # Plateau / target checks after each iteration
            if trace.score_events:
                latest = trace.score_events[-1].overall
                recent_scores.append(latest)

                if latest >= traits.target_score:
                    trace.stop_reason = f"target_score_reached ({latest:.3f})"
                    break

                if len(recent_scores) >= traits.plateau_window:
                    window = recent_scores[-traits.plateau_window:]
                    if max(window) - min(window) < traits.plateau_min_delta:
                        trace.stop_reason = f"plateau ({window})"
                        break
        else:
            trace.stop_reason = "max_iterations"

    # ── Tool dispatch ──────────────────────────────────────────────

    def _call_tool(
        self,
        name: str,
        args: dict[str, Any],
        reference: dict[str, Any],
        ws: Path,
        traits: AgentTraits,
        trace: HarnessTrace,
        iteration: int,
        em: Emitter,
    ) -> Any:
        b = self._blender
        assert b is not None

        if name == "image_to_pointcloud":
            ply_path = self._pointcloud.convert(args["image_path"], args["output_ply_path"])
            em.artifact("ply", ply_path)
            return ply_path

        if name == "blender_import_file":
            return b.import_file(args["path"], args.get("file_format"))
        if name == "blender_export_file":
            return b.export_file(args["path"], args.get("file_format"), args.get("object_names"))
        if name == "blender_execute_python":
            return b.execute_python(args["code"])
        if name == "blender_get_scene_info":
            return b.get_scene_info()
        if name == "blender_smooth_mesh":
            return b.smooth_mesh(args["object_name"], args.get("iterations", 5), args.get("factor", 0.5))
        if name == "blender_apply_subdivision":
            return b.apply_subdivision(args["object_name"], args.get("levels", 2))
        if name == "blender_unwrap_uv":
            return b.unwrap_uv(args["object_name"], args.get("method", "SMART_PROJECT"))
        if name == "blender_get_topology_stats":
            return b.get_topology_stats(args["object_name"])
        if name == "blender_get_vertex_positions":
            return b.get_vertex_positions(args["object_name"])
        if name == "blender_render":
            return b.render(
                args["output_path"],
                args.get("camera_angles", traits.render_angles),
                tuple(args["resolution"]) if "resolution" in args else traits.render_resolution,
                args.get("engine", "CYCLES"),
            )
        if name == "blender_get_depth_map":
            return b.get_depth_map(
                args["output_path"],
                tuple(args["resolution"]) if "resolution" in args else traits.render_resolution,
            )
        if name == "blender_set_material":
            return b.set_material(args["object_name"], args["texture_path"], args.get("mapping", "UV"))

        if name == "generate_texture":
            return self._texture_gen.generate_from_prompt(
                args["prompt"], args["output_path"],
                args.get("negative_prompt", "seam, low quality, blurry"),
                args.get("size"),
                args.get("texture_type", "albedo"),
            )
        if name == "generate_texture_from_reference":
            return self._texture_gen.generate_from_reference(
                args["reference_image_path"], args["output_path"],
                args.get("prompt", ""), args.get("strength", 0.6),
                args.get("texture_type", "albedo"),
            )

        if name == "evaluate_render":
            payload = RenderPayload(
                render_images=args["render_image_paths"],
                depth_map_path=args.get("depth_map_path"),
                ply_path=args.get("ply_path"),
            )
            obj = args.get("object_name")
            if obj:
                try:
                    payload.topology_stats   = b.get_topology_stats(obj)
                    payload.vertex_positions = b.get_vertex_positions(obj)
                except Exception:
                    pass

            ev = self.evaluator.evaluate(payload, reference)
            trace.final_evaluation = ev

            se = ScoreEvent(
                iteration=iteration,
                overall=ev.overall_score,
                visual=ev.visual_fidelity,
                topology=ev.topology_quality,
                depth=ev.depth_alignment,
                vertices=ev.vertex_accuracy,
                feedback=ev.feedback,
            )
            trace.score_events.append(se)
            em.score(
                iteration=iteration,
                overall=ev.overall_score,
                visual=ev.visual_fidelity,
                topology=ev.topology_quality,
                depth=ev.depth_alignment,
                vertices=ev.vertex_accuracy,
                feedback=ev.feedback,
            )

            # Return weighted breakdown so the model can see trait-adjusted priorities
            fa = FeedbackAccessor(ev)
            return {
                "overall_score":    ev.overall_score,
                "weighted_scores": {
                    "topology": round(ev.topology_quality * traits.weight_topology, 3),
                    "texture":  round(ev.visual_fidelity  * traits.weight_texture,  3),
                    "shape":    round(ev.visual_fidelity  * traits.weight_shape,     3),
                    "depth":    round(ev.depth_alignment  * traits.weight_depth,     3),
                },
                "priority_feedback": [i.raw for i in fa.priority_order()[:5]],
                "target_score":  traits.target_score,
                "iterations_left": traits.max_iterations - iteration - 1,
            }

        raise ValueError(f"Unknown tool: {name}")

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _initial_message(goal: str, reference: dict[str, Any], ws: Path) -> str:
        parts = [f"Goal: {goal}", f"Workspace: {ws}"]
        if reference.get("image_path"):
            parts.append(f"Reference image: {reference['image_path']}")
        if reference.get("reference_mesh_path"):
            parts.append(f"Reference mesh: {reference['reference_mesh_path']}")
        parts.append("Begin.")
        return "\n".join(parts)

    def _build_client(self, api_key_override: str = "") -> OpenAI:
        if self.cfg.llm_backend == "openrouter":
            key = api_key_override or _secrets.get("OPENROUTER_API_KEY")
            if not key:
                raise EnvironmentError("OPENROUTER_API_KEY is not set")
            return OpenAI(
                api_key=key,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": "https://github.com/blender-agent",
                    "X-Title": "BlenderAgent",
                },
            )
        key = api_key_override or _secrets.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        return OpenAI(api_key=key)

    @staticmethod
    def _preflight(mcp: MCPConfig) -> None:
        import socket as _socket
        s = _socket.socket()
        s.settimeout(3)
        try:
            s.connect((mcp.host, mcp.port))
            s.close()
        except OSError:
            raise RuntimeError(
                f"Blender MCP server not reachable at {mcp.host}:{mcp.port}. "
                "In Blender: N-panel → MCP tab → Start MCP Server."
            )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
