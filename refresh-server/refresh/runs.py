"""Runs: one uipack reconstruction = one lineage search scope over one reference image."""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from blender_agent import solidity
from blender_agent.reference_guard import ReferenceGuard
from blender_mcp_connector import BlenderMCPConnector, MCPConfig
from lineage.search import search
from lineage.store import log

from .config import settings
from .stage import stage_spec, task_hash
from .workbench import BLENDER_LOCK, BlenderWorkbench


@dataclass
class RunContext:
    run_id: str
    project_id: str
    dir: Path
    reference: str
    extra_views: list[str]
    stage: dict[str, Any]
    task_hash: str
    created: float = field(default_factory=time.time)

    @property
    def scope(self) -> str:
        return self.run_id

    def save(self) -> None:
        d = asdict(self)
        d["dir"] = str(self.dir)
        (self.dir / "run.json").write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, run_dir: Path) -> "RunContext":
        d = json.loads((run_dir / "run.json").read_text())
        d["dir"] = Path(d["dir"])
        return cls(**d)


class RunManager:
    def __init__(self, db: Any):
        self.db = db
        self.connector = BlenderMCPConnector(MCPConfig())
        self.runs: dict[str, RunContext] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.stops: dict[str, threading.Event] = {}

    # ── lifecycle ─────────────────────────────────────────────────

    def get(self, run_id: str) -> RunContext:
        if run_id not in self.runs:
            run_dir = settings.workspace / "runs" / run_id
            if not (run_dir / "run.json").exists():
                raise KeyError(run_id)
            self.runs[run_id] = RunContext.load(run_dir)
        return self.runs[run_id]

    def create(self, project_id: str, images: list[tuple[str, bytes]]) -> RunContext:
        if not images:
            raise ValueError("at least one reference image is required")
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_dir = settings.workspace / "runs" / run_id
        (run_dir / "refs").mkdir(parents=True, exist_ok=True)
        paths = []
        for i, (name, data) in enumerate(images):
            p = run_dir / "refs" / f"ref_{i:02d}{Path(name).suffix.lower() or '.png'}"
            p.write_bytes(data)
            paths.append(str(p))
        spec = stage_spec(paths[0], settings.stage_resolution)
        ctx = RunContext(run_id=run_id, project_id=project_id, dir=run_dir, reference=paths[0],
                         extra_views=paths[1:], stage=spec, task_hash=task_hash(paths[0], spec))
        ctx.save()
        self.runs[run_id] = ctx
        return ctx

    def start(self, ctx: RunContext) -> None:
        stop = threading.Event()
        self.stops[ctx.run_id] = stop
        t = threading.Thread(target=self._search, args=(ctx, stop), daemon=True, name=f"search-{ctx.run_id}")
        self.threads[ctx.run_id] = t
        t.start()

    def stop(self, run_id: str) -> bool:
        ev = self.stops.get(run_id)
        if ev:
            ev.set()
        return ev is not None

    def _search(self, ctx: RunContext, stop: threading.Event) -> None:
        log(self.db, ctx.scope, "run_started", reference=settings.url(ctx.reference),
            extra_views=[settings.url(p) for p in ctx.extra_views], stage=ctx.stage)
        if settings.agent_backend == "claude_code" or settings.generator_backend == "claude_code":
            from blender_agent.claude_code_loop import auth_status

            st = auth_status()
            if not st.get("loggedIn"):
                log(self.db, ctx.scope, "search_failed", reason=(
                    "Claude Code is not logged in for the process running refresh-server "
                    f"({st.get('error') or st.get('authMethod', 'no credentials')}). Run `claude` then /login "
                    "(or `claude auth login`) in that terminal, or put a `claude setup-token` token in .env as "
                    "CLAUDE_CODE_OAUTH_TOKEN, then restart refresh-server."))
                return
        try:
            with BLENDER_LOCK:
                self.connector.setup_stage(ctx.stage)
                self.connector.restore(None)
        except Exception as exc:
            log(self.db, ctx.scope, "search_failed", reason=f"Blender is not reachable: {exc}")
            return
        try:
            search(scope=ctx.scope, memory=settings.memory, generations=settings.generations, k=settings.k,
                   workbench=BlenderWorkbench(ctx, self.connector, self.db), task_id="recon",
                   task_input=ctx.reference, seed=settings.seed, model=settings.generator_model, db=self.db,
                   stop_event=stop)
        except Exception as exc:
            log(self.db, ctx.scope, "search_failed", reason=f"{type(exc).__name__}: {exc}")

    # ── uipack event stream ───────────────────────────────────────

    def translator(self):
        """lineage event -> uipack RunUpdate payloads (status / metric / model / error, then [DONE])."""
        state = {"best": -1.0, "best_glb": None, "done": False}

        def status(text: str) -> dict[str, Any]:
            return {"type": "status", "status": text}

        def tr(e: dict[str, Any]) -> list[Any]:
            if state["done"]:
                return []
            kind = e.get("kind")
            if kind in ("run_started", "search_started", "resume"):
                return [status("Reading reference views")]
            if kind == "structure_started":
                roles = ", ".join(n.get("role", "?") for n in e.get("nodes", []))
                return [status(f"Agent handoff: new team of {len(e.get('nodes', []))} ({roles})")]
            if kind == "node_started":
                return [status(f"Agent {e.get('node_id')} generating geometry")]
            if kind == "cache_hit":
                return [status(f"Agent {e.get('node_id')} reused from memory")]
            if kind in ("blocked", "near_dup"):
                return [status("Skipped a team already tried")]
            if kind == "structure_failed":
                return [status("A team failed; trying the next")]
            if kind == "eval":
                out: list[Any] = [status("Comparing model similarity"),
                                  {"type": "metric", "similarity": round(100 * float(e.get("fitness") or 0), 1)}]
                if e.get("glb") and float(e.get("fitness") or 0) > state["best"]:
                    state["best"], state["best_glb"] = float(e["fitness"]), e["glb"]
                    out.append({"type": "model", "modelId": (e.get("structure_hash") or "")[:12],
                                "modelUrl": e["glb"], "status": "Model updated"})
                return out
            if kind == "generation_done":
                best = e.get("best")
                return [status(f"Generation {e.get('gen')} done" + (f", best {100 * best:.0f}%" if best else ""))]
            if kind == "refined":
                return [{"type": "model", "modelId": e.get("model_id"), "modelUrl": e.get("glb"),
                         "status": "Refinement complete"}]
            if kind == "search_failed":
                state["done"] = True
                return [{"type": "error", "message": e.get("reason", "Reconstruction failed")}, "[DONE]"]
            if kind in ("search_done", "stopped"):
                state["done"] = True
                out = []
                glb = e.get("glb") or state["best_glb"]
                if glb and glb != state["best_glb"]:
                    out.append({"type": "model", "modelId": (e.get("best_structure") or "")[:12], "modelUrl": glb})
                return out + ["[DONE]"]
            return []

        return tr

    # ── sculpt refinement ─────────────────────────────────────────

    def refine(self, run_id: str, model_id: str, selection: dict[str, Any]) -> dict[str, Any]:
        """One refiner BlenderAgent on the best structure's model, guided by the pinch-sculpt selection."""
        from blender_agent import AgentConfig, AgentTraits, BlenderAgent, CollectionSink, Emitter, EvaluatorClient
        from blender_agent.texture_gen import TextureGenConfig
        from render_eval import EvalConfig
        from render_eval.project import record_run

        ctx = self.get(run_id)
        best = self.db.structures.find_one({"scope": ctx.scope, "blend": {"$exists": True}},
                                           sort=[("fitness.mean", -1)])
        prev = self.db.refinements.find_one({"scope": ctx.scope}, sort=[("t", -1)])
        base_blend = (prev or {}).get("blend") or (best or {}).get("blend")
        n = self.db.refinements.count_documents({"scope": ctx.scope}) + 1
        rdir = ctx.dir / "refinements" / f"r{n:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        shot = None
        if isinstance(selection.get("screenshot"), str) and "," in selection["screenshot"]:
            shot = rdir / "selection.png"
            shot.write_bytes(base64.b64decode(selection["screenshot"].split(",", 1)[1]))
        brief = describe_selection(selection)
        traits = AgentTraits(
            persona=("Your role: refiner. The user pinch-sculpted a region of the current model in the web "
                     f"viewer and wants that region improved. {brief} The selection screenshot shows the "
                     "region; match it to the reference photo. Change only that region unless the score "
                     "shows a problem elsewhere."),
            allowed_tools=["blender_execute_python", "blender_get_vertex_positions", "blender_smooth_mesh",
                           "blender_apply_subdivision", "blender_render", "blender_set_material", "edit_image"],
            max_iterations=settings.agent_max_iterations, target_score=settings.agent_target_score,
        )
        with BLENDER_LOCK:
            self.connector.setup_stage(ctx.stage)
            self.connector.restore(base_blend)
            sink = CollectionSink(self.db.agent_events, scope=ctx.scope, run_id=run_id, structure_hash=f"refine-{n}",
                                  node_id="refiner", role="refiner", gen=-1)
            emitter = Emitter(sink, str(rdir / "agent.log"))
            agent = BlenderAgent(
                evaluator=EvaluatorClient.from_render_eval(fast=True),
                config=AgentConfig(model=settings.agent_model, llm_backend=settings.agent_backend,
                                   effort=settings.agent_effort, mcp=self.connector.config,
                                   texture=TextureGenConfig(backend=settings.imagegen_backend), workspace=str(rdir)),
            )
            try:
                agent.run(goal="Refine the selected region of the model.",
                          reference={"image_path": ctx.reference, "extra_views": [str(shot)] if shot else []},
                          traits=traits, workspace=str(rdir), emitter=emitter, connector=self.connector)
            finally:
                emitter.close()
            blend, png, glb = rdir / "model.blend", rdir / "stage.png", rdir / "model.glb"
            self.connector.snapshot(str(blend))
            self.connector.setup_stage(ctx.stage)
            self.connector.render_stage(str(png))
            self.connector.export_glb(str(glb))
            solid = solidity.measure(self.connector, rdir / "turntable")
            pasted = ReferenceGuard({"image_path": ctx.reference,
                                     "extra_views": ctx.extra_views}).scene_violations(self.connector.scene_images())
        res = record_run(ctx.reference, str(png), rdir / "eval", EvalConfig(), label=f"refinement {n}",
                         meta={"scope": ctx.scope, "refines": model_id})
        new_id = f"refine-{n}"
        doc = {"scope": ctx.scope, "t": time.time(), "model_id": new_id, "parent_model_id": model_id,
               "blend": str(blend), "glb": settings.url(glb), "render": settings.url(png),
               "composite": 0.0 if pasted else (res["entry"].get("composite") or 0.0) * solidity.fitness_factor(solid["solidity"]),
               "front_match": res["entry"].get("composite"), "solidity": solid["solidity"], "disqualified": pasted,
               "critique": res["critique"], "selection_brief": brief}
        self.db.refinements.insert_one(dict(doc))
        log(self.db, ctx.scope, "refined", model_id=new_id, glb=doc["glb"], composite=doc["composite"])
        return {"modelId": new_id, "modelUrl": doc["glb"], "composite": doc["composite"]}


def describe_selection(selection: dict[str, Any]) -> str:
    """Summarise uipack's selection packet in Blender coordinates (glTF Y-up -> Blender Z-up: x, -z, y)."""
    def to_blender(p: list[float]) -> list[float]:
        x, y, z = (list(p) + [0, 0, 0])[:3]
        return [round(x, 3), round(-z, 3), round(y, 3)]

    pts = [to_blender(p) for p in selection.get("points") or []]
    parts = []
    if pts:
        c = [round(sum(p[i] for p in pts) / len(pts), 3) for i in range(3)]
        parts.append(f"The selected region is centred near Blender coordinates {c} ({len(pts)} sample points).")
    edits = selection.get("geometryEdits") or []
    for e in edits[:4]:
        verts = e.get("vertices") or []
        if not verts:
            continue
        d = [sum((v["position"][i] - v["original"][i]) for v in verts) / len(verts) for i in range(3)]
        parts.append(f"On mesh '{e.get('meshName', '?')}' the user moved {len(verts)} vertices by about "
                     f"{to_blender(d)} (Blender axes); reproduce that intent cleanly.")
    if selection.get("faces"):
        parts.append(f"{len(selection['faces'])} faces were selected.")
    return " ".join(parts) or "No geometry details were sent; use the screenshot."
