"""BlenderWorkbench: lineage's workbench contract implemented with BlenderAgent + Blender + render-eval.

lineage calls, per structure:
  run_node(genome, inputs, ctx)   one lineage node = one BlenderAgent run (genome -> AgentTraits)
  evaluate(task_id, s_hash, sinks) full render-eval of the stage render -> fitness + vectors

Nodes run one at a time in the single live Blender (lock). The model moves between nodes as
.blend snapshots in the run's workspace; output_hash is the snapshot's content hash, which is
what lineage's node cache keys on.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from blender_agent import AgentConfig, AgentTraits, BlenderAgent, CollectionSink, Emitter, EvaluatorClient
from blender_agent import solidity
from blender_agent.reference_guard import ReferenceGuard
from blender_agent.texture_gen import TextureGenConfig
from blender_agent.tools import registry
from lineage.hashing import content_hash

from .config import settings

if TYPE_CHECKING:
    from blender_mcp_connector import BlenderMCPConnector

    from .runs import RunContext

BLENDER_LOCK = threading.Lock()  # one live Blender: nodes and evals never overlap, across runs too

TASK_BRIEF = (
    "Rebuild the object in the reference photo as a complete 3-D model in a live Blender scene, so that its "
    "render from a fixed stage camera matches the photo (shape, proportions, parts, colour, materials) and it "
    "is a real, closed object from every side (a solidity check looks all around it; a relief or a half-shell "
    "scores low). The photo must never be pasted, projected or turned into geometry. Agents in "
    "a structure run in order and each continues the previous agent's scene; the first agent starts from an "
    "empty stage, so it should build the geometry (planner or geometry). Useful roles: planner (plans "
    "and blocks out the main masses, may make reference images), preprocessor (works out parts and proportions "
    "and builds a rough base mesh), geometry (builds and shapes meshes), refiner (fixes proportions and details the "
    "score points at), texturer (materials and textures with imagegen), verifier (checks and fixes "
    "whatever scores worst). Every agent can inspect the scene and score its work."
)

ROLE_GUIDE = {
    "planner": "Plan the model: identify the main parts and block them out with simple primitives at the right "
               "positions and proportions. You may generate helper images with generate_image.",
    "preprocessor": "Prepare the ground for the others: study the reference, work out the object's parts and "
                    "proportions (including the sides the photo can't show), and build a clean rough base mesh "
                    "the next agent can refine.",
    "geometry": "Build and shape the geometry so the silhouette, depth and proportions match the reference.",
    "refiner": "Refine the existing model: fix the parts the evaluation says are weakest (outline, depth, "
               "normals, interior edges) without rebuilding what already works.",
    "texturer": "Give the model its colours and materials (materials, generate_texture, retexture_uv) "
                "once the shape is right.",
    "verifier": "Evaluate the current model, find its weakest aspect and fix it.",
}

ROLE_WEIGHTS = {
    "geometry": dict(weight_shape=1.5, weight_depth=1.0, weight_texture=0.3),
    "refiner": dict(weight_shape=1.2, weight_depth=1.2),
    "texturer": dict(weight_texture=1.5, weight_shape=0.5),
    "planner": dict(weight_shape=1.3, weight_texture=0.3),
}


def traits_from_genome(genome: dict[str, Any]) -> AgentTraits:
    role = genome.get("role", "agent")
    params = genome.get("params") or {}
    return AgentTraits(
        persona=f"Your role: {role}. {ROLE_GUIDE.get(role, '')}\nYour brief: {genome.get('brief', '')}",
        allowed_tools=list(genome.get("tools") or []) or None,
        max_iterations=max(2, min(settings.agent_max_iterations, int(params.get("max_calls", 6)))),
        target_score=settings.agent_target_score,
        **ROLE_WEIGHTS.get(role, {}),
    )


class BlenderWorkbench:
    version = "blender-agent-1"
    task_brief = TASK_BRIEF

    def __init__(self, run: "RunContext", connector: "BlenderMCPConnector", db: Any):
        self.run = run
        self.connector = connector
        self.db = db
        self.eval_version = f"render-eval-1+solidity-1:{run.task_hash[:8]}"

    # ── lineage contract ──────────────────────────────────────────

    def tools(self) -> list[dict[str, Any]]:
        return registry()

    def task_artifact(self, task_id: str, task_input: str) -> dict[str, Any]:
        return {"ref": "task:empty-stage", "hash": self.run.task_hash, "chain": [],
                "summary": "empty stage scene; the reference image is the target"}

    def run_node(self, genome: dict[str, Any], inputs: list[dict[str, Any]], ctx: dict[str, Any]) -> dict[str, Any]:
        node_dir = self.run.dir / "nodes" / ctx["input_key"][:16]
        node_dir.mkdir(parents=True, exist_ok=True)
        role = genome.get("role", "agent")
        info = {"role": role, "structure_hash": ctx["structure_hash"], "node_id": ctx["node_id"], "gen": ctx["gen"],
                "started": time.time()}
        (node_dir / "node.json").write_text(json.dumps(info))  # lets the dashboard label live renders
        for sub_dir in ("scripts", "debug", "renders", "blend"):
            (node_dir / sub_dir).mkdir(exist_ok=True)
        # a readable name for the agent's folder: agents/g0_1a2b3c4d_n1_geometry -> nodes/<cache key>
        alias = self.run.dir / "agents" / f"g{ctx['gen']}_{ctx['structure_hash'][:8]}_{ctx['node_id']}_{role}"
        try:
            alias.parent.mkdir(exist_ok=True)
            if not alias.exists():
                alias.symlink_to(Path("..") / "nodes" / node_dir.name, target_is_directory=True)
        except OSError:
            pass
        with BLENDER_LOCK:
            self._load_inputs(inputs)
            try:  # the scene this agent was handed, openable in Blender
                self.connector.save_blend(str(node_dir / "blend" / "start.blend"))
            except Exception:
                pass
            ctx["log"]("agent_started", role=role, tools=genome.get("tools", []))
            upstream = "; ".join(i.get("summary", "") for i in inputs if not i["ref"].startswith("task:"))
            reference = {"image_path": self.run.reference, "extra_views": self.run.extra_views,
                         "brief_extra": f"Work so far from earlier agents: {upstream}" if upstream else ""}
            sink = CollectionSink(self.db.agent_events, scope=self.run.scope, run_id=self.run.run_id,
                                  structure_hash=ctx["structure_hash"], node_id=ctx["node_id"], role=role,
                                  gen=ctx["gen"])
            emitter = Emitter(sink, str(node_dir / "agent.log"))
            agent = BlenderAgent(
                evaluator=EvaluatorClient.from_render_eval(fast=True),
                config=AgentConfig(model=settings.agent_model, llm_backend=settings.agent_backend,
                                   effort=settings.agent_effort, mcp=self.connector.config,
                                   texture=TextureGenConfig(backend=settings.imagegen_backend),
                                   workspace=str(node_dir)),
            )
            error, trace = None, None
            try:
                _, trace = agent.run(goal=f"You are the {role} agent of a modelling team.", reference=reference,
                                     traits=traits_from_genome(genome), workspace=str(node_dir), emitter=emitter,
                                     connector=self.connector)
            except ConnectionError:
                raise
            except Exception as exc:  # keep whatever the agent built; the structure's eval will judge it
                error = f"{type(exc).__name__}: {exc}"
                if _is_auth_error(exc):  # no working LLM credentials: every agent would fail, stop the search
                    raise ConnectionError(f"Agent model authentication failed ({settings.agent_backend}): {exc}") from exc
            finally:
                emitter.close()
            snap = node_dir / "node.blend"  # hand-off to the next agent (model objects only)
            self.connector.snapshot(str(snap))
            try:  # the full scene this agent left, openable in Blender
                self.connector.save_blend(str(node_dir / "blend" / "end.blend"))
            except Exception:
                pass
            try:  # work-in-progress model for the dashboard, before the team is scored
                self.connector.export_glb(str(node_dir / "node.glb"))
                (node_dir / "node.json").write_text(json.dumps({**info, "t": time.time()}))
            except Exception:
                pass
        best = max(trace.score_progression(), default=None) if trace else None
        summary = f"{role} agent ({trace.stop_reason if trace else 'failed'})"
        if best is not None:
            summary += f", best inner score {best:.2f}"
        if trace and trace.final_text:
            summary += f". {' '.join(trace.final_text.split())[:500]}"
        if error:
            summary += f", error: {error[:200]}"
        return {
            "output_ref": str(snap),
            "output_hash": content_hash(snap.read_bytes()),
            "chain": [role],
            "summary": summary,
            "cost_usd": trace.cost_usd if trace else 0.0,
            "error": error,
            "note": trace.stop_reason if trace else error,
            "calls": [{"tool_id": e.tool, "args": _short(e.args), "error": e.error,
                       "summary": _short(e.result)} for e in (trace.tool_events if trace else [])],
        }

    def evaluate(self, task_id: str, structure_hash: str, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        from render_eval import EvalConfig
        from render_eval.project import record_run

        sdir = self.run.dir / "structures" / structure_hash[:16]
        guard = ReferenceGuard({"image_path": self.run.reference, "extra_views": self.run.extra_views})
        best: dict[str, Any] | None = None
        for i, art in enumerate(artifacts):
            png, glb = sdir / f"stage_{i}.png", sdir / f"model_{i}.glb"
            with BLENDER_LOCK:
                self.connector.setup_stage(self.run.stage)
                self.connector.restore(art["ref"])
                pasted = guard.scene_violations(self.connector.scene_images())
                self.connector.render_stage(str(png))
                self.connector.export_glb(str(glb))
                solid = solidity.measure(self.connector, sdir / f"turntable_{i}")
            res = record_run(self.run.reference, str(png), sdir / "eval", EvalConfig(),
                             label=f"{structure_hash[:8]} sink {i}",
                             meta={"scope": self.run.scope, "structure_hash": structure_hash})
            report, vec = res["report"], res["vector"]
            judge = report.results.get("judge")
            front = float(report.composite or 0.0)
            # Front match x solidity: a relief or an open half-shell can't win on the stage view alone,
            # and a scene that loads the reference photo is disqualified outright.
            fitness = 0.0 if pasted else front * solidity.fitness_factor(solid["solidity"])
            steps = {k: r.score for k, r in report.results.items() if r.ok} | {"solidity": solid["solidity"]}
            critique = res["critique"]
            if pasted:
                critique = (f"Disqualified: the scene loads the reference photo ({', '.join(pasted)}); the photo was "
                            "pasted or projected instead of modelled.\n\n" + critique)
            elif solid["feedback"]:
                critique = " ".join(solid["feedback"]) + "\n\n" + critique
            cand = {
                "fitness": fitness,
                "front_match": front,
                "solidity": {k: solid[k] for k in ("solidity", "closure", "thickness")},
                "disqualified": pasted,
                "turntable": settings.url(solid["sheet"]) if solid["sheet"] else None,
                "metrics": steps,
                "scores": steps,
                "per_node": {},
                "eval_version": self.eval_version,
                "cost_usd": float(((judge.details.get("usage") or {}).get("cost") or 0.0)) if judge and judge.ok else 0.0,
                "critique": critique,
                "critique_fixes": solid["feedback"][:2] + (list(judge.details.get("top_fixes") or []) if judge and judge.ok else []),
                "score_vec": [0.0 if s is None else s for s in vec.scores],
                "outcome_vec": [round(float(x), 5) for x in vec.flat()],
                "eval_dir": str(res["latest_dir"]),
                "render": settings.url(png),
                "glb": settings.url(glb),
                "overview": settings.url(Path(res["latest_dir"]) / "overview.png"),
                "blend": art["ref"],
            }
            if best is None or cand["fitness"] > best["fitness"]:
                best = cand
        if best is None:
            raise ValueError("structure produced no outputs to evaluate")
        return best

    # ── helpers ───────────────────────────────────────────────────

    def _load_inputs(self, inputs: list[dict[str, Any]]) -> None:
        """Fresh stage + the upstream agents' snapshots (several parents are merged)."""
        self.connector.setup_stage(self.run.stage)
        refs = [i["ref"] for i in inputs if not i["ref"].startswith("task:")]
        self.connector.restore(refs[0] if refs else None)
        for extra in refs[1:]:
            self.connector.restore(extra, clear=False)


def _is_auth_error(exc: Exception) -> bool:
    name, text = type(exc).__name__.lower(), str(exc).lower()
    return ("authentication" in name or "permissiondenied" in name or "api_key" in text
            or "authentication method" in text or "is not set" in text and "api_key" in text.replace(" ", "_"))


def _short(value: Any, limit: int = 300) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"
