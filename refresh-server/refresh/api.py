"""Refresh API: the backend uipack talks to.

uipack contract (see uipack/README.md):
  POST /api/reconstructions                 multipart projectId + images -> {runId, eventsUrl}
  GET  /api/reconstructions/{runId}/events  SSE: status / metric / model / error, then [DONE]
  POST /api/reconstructions/{runId}/refine  {modelId, selection} -> {modelId, modelUrl}

Dashboard (search view):
  GET  /api/runs, /api/runs/{runId}, /api/runs/{runId}/structures, /api/runs/{runId}/structures/{hash},
       /api/runs/{runId}/agent-events, /api/runs/{runId}/vectors, POST /api/reconstructions/{runId}/stop
  lineage's own endpoints: /api/state, /api/stream, /api/module, /api/lineage, /api/scopes, /lineage-ui
  files: /files/... (renders, GLBs, eval overviews from the workspace)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from lineage.server import event_stream, router as lineage_router, set_db
from lineage.store import ensure_indexes, get_db, similar_outcomes

from .config import settings
from .runs import RunManager

settings.workspace.mkdir(parents=True, exist_ok=True)
db = get_db()
ensure_indexes(db)
set_db(db)
manager = RunManager(db)

app = FastAPI(title="Refresh API")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
app.include_router(lineage_router)
app.mount("/files", StaticFiles(directory=str(settings.workspace)), name="files")

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _clean(d: dict[str, Any] | None, drop: tuple[str, ...] = ("_id",)) -> dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if k not in drop}


# ── uipack contract ───────────────────────────────────────────────


@app.post("/api/reconstructions", status_code=202)
async def start_reconstruction(projectId: Annotated[str, Form()], images: Annotated[list[UploadFile], File()]):
    files = []
    for f in images:
        if not (f.content_type or "").startswith("image/"):
            raise HTTPException(400, f"{f.filename} is not an image")
        files.append((f.filename or "reference.png", await f.read()))
    try:
        ctx = manager.create(projectId, files)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    manager.start(ctx)
    return {"runId": ctx.run_id, "eventsUrl": f"/api/reconstructions/{ctx.run_id}/events"}


@app.get("/api/reconstructions/{run_id}/events")
async def reconstruction_events(run_id: str):
    _require(run_id)
    return StreamingResponse(event_stream(run_id, 0.0, manager.translator()), media_type="text/event-stream",
                             headers=SSE_HEADERS)


@app.post("/api/reconstructions/{run_id}/refine")
def refine(run_id: str, body: dict[str, Any] = Body(...)):
    _require(run_id)
    try:
        return manager.refine(run_id, body.get("modelId") or "", body.get("selection") or {})
    except ConnectionError as exc:
        raise HTTPException(503, f"Blender is not reachable: {exc}") from exc


@app.post("/api/reconstructions/{run_id}/stop")
def stop(run_id: str):
    _require(run_id)
    return {"stopping": manager.stop(run_id)}


@app.get("/")
def index():
    """This is the API; the dashboard is uipack (http://localhost:3000)."""
    return {
        "service": "Refresh API",
        "dashboard": "http://localhost:3000",
        "health": f"{settings.public_url}/api/health",
        "lineage_live_view": f"{settings.public_url}/lineage-ui",
        "api_docs": f"{settings.public_url}/docs",
    }


# ── dashboard reads ───────────────────────────────────────────────


@app.get("/api/health")
def health():
    import os

    agents = _agents_auth()
    return {
        "blender": manager.connector.ping(),
        "database": type(db).__module__.split(".")[0],  # pymongo (Atlas) or mongomock
        "keys": {k: bool(os.getenv(k)) for k in ("OPENROUTER_API_KEY", "MONGODB_URI", "CLAUDE_CODE_OAUTH_TOKEN")},
        "agent_backend": settings.agent_backend,
        "agents_ready": bool(agents.get("loggedIn")),
        "agents_auth": agents,
        "agent_model": settings.agent_model or "cli default",
        "generator_backend": settings.generator_backend,
        "generator_model": settings.generator_model or "cli default",
    }


_AUTH_CACHE: dict[str, Any] = {}


def _agents_auth() -> dict[str, Any]:
    """Can the agents reach their model? claude_code: `claude auth status` (cached a minute); else the API key."""
    import os
    import time

    if settings.agent_backend != "claude_code":
        key = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}.get(settings.agent_backend,
                                                                                  "ANTHROPIC_API_KEY")
        return {"loggedIn": bool(os.getenv(key)), "authMethod": key}
    if time.time() - _AUTH_CACHE.get("t", 0.0) > 60:
        from blender_agent.claude_code_loop import auth_status

        st = auth_status()
        _AUTH_CACHE.update(t=time.time(), status={k: st[k] for k in ("loggedIn", "authMethod", "apiProvider",
                                                                     "subscriptionType", "error", "bin") if k in st})
    return _AUTH_CACHE["status"]


@app.get("/api/runs")
def runs():
    root = settings.workspace / "runs"
    out = []
    for d in sorted(root.glob("run_*"), reverse=True) if root.exists() else []:
        try:
            out.append(run_summary(d.name))
        except HTTPException:
            continue
    return out


@app.get("/api/runs/{run_id}")
def run_summary(run_id: str):
    ctx = _require(run_id)
    q = {"scope": run_id}
    best = db.structures.find_one({**q, "fitness.n": {"$gte": 1}}, sort=[("fitness.mean", -1)])
    last = db.events.find_one(q, sort=[("t", -1)])
    kinds = {k: db.events.count_documents({**q, "kind": k})
             for k in ("eval", "blocked", "near_dup", "cache_hit", "structure_failed")}
    done = last is not None and last.get("kind") in ("search_done", "stopped", "search_failed")
    running = manager.threads.get(run_id) is not None and manager.threads[run_id].is_alive()
    status = (last or {}).get("kind", "queued") if done else ("running" if running else
                                                              "interrupted" if last else "queued")
    cur = db.agent_events.find_one(q, sort=[("ts", -1)]) if running else None
    return {
        "runId": run_id, "projectId": ctx.project_id, "created": ctx.created,
        "reference": settings.url(ctx.reference), "extraViews": [settings.url(p) for p in ctx.extra_views],
        "status": status,
        "statusReason": (last or {}).get("reason") if status == "search_failed" else None,
        "current": {"role": cur.get("role"), "event": cur.get("event"), "tool": cur.get("tool"),
                    "structureHash": cur.get("structure_hash"), "generation": cur.get("gen"), "ts": cur.get("ts")}
                   if cur else None,
        "running": running,
        "latestModel": _latest_model(ctx),
        "best": _structure_card(best) if best else None,
        "counts": {**kinds, "agent_calls": db.agent_events.count_documents({**q, "event": "tool_call"})},
        "generations": settings.generations, "k": settings.k,
    }


@app.get("/api/runs/{run_id}/structures")
def structures(run_id: str):
    _require(run_id)
    docs = db.structures.find({"scope": run_id}).sort([("generation", 1), ("created", 1)])
    return [_structure_card(d) for d in docs]


@app.get("/api/runs/{run_id}/structures/{structure_hash}")
def structure_detail(run_id: str, structure_hash: str):
    _require(run_id)
    d = db.structures.find_one({"scope": run_id, "structure_hash": structure_hash})
    if not d:
        raise HTTPException(404, "structure not found")
    card = _structure_card(d)
    card["critique"] = d.get("critique", "")
    card["modules"] = [_clean(m, ("_id", "calls")) | {"calls": len(m.get("calls", []))}
                       for m in db.modules.find({"scope": run_id, "structure_hash": structure_hash})]
    if d.get("outcome_vec"):
        card["similar"] = [{"structureHash": s["structure_hash"], "similarity": round(s["similarity"], 3),
                            "fitness": (s.get("fitness") or {}).get("mean"),
                            "roles": [n.get("role") for n in s.get("nodes", [])]}
                           for s in similar_outcomes(db, run_id, d["outcome_vec"], k=3, exclude=structure_hash)]
    return card


@app.get("/api/runs/{run_id}/renders")
def renders(run_id: str, structure_hash: str | None = None, limit: int = 80):
    """Every image the agents have produced so far, newest first, before anything is scored:
    preview (quick stage view after each scene change), stage (the scored view, with its score),
    turntable (8 sides, back faces red), render (the agent's own camera angles), image (generated)."""
    import json

    ctx = _require(run_id)
    scores = {}
    for e in db.agent_events.find({"scope": run_id, "event": "artifact", "kind": {"$in": ["stage_render", "turntable"]}}):
        scores[str(Path(e.get("path", "")).resolve())] = {k: e.get(k) for k in ("overall", "front_match", "solidity")}
    out = []
    for node_dir in ctx.dir.glob("nodes/*"):
        try:
            info = json.loads((node_dir / "node.json").read_text())
        except (OSError, ValueError):
            info = {}
        if structure_hash and info.get("structure_hash") != structure_hash:
            continue
        for p in node_dir.rglob("*.png"):
            rel = p.relative_to(node_dir).as_posix()
            if "/view_" in f"/{rel}":
                continue  # turntable tiles; the strip is listed instead
            kind = ("preview" if rel.startswith("previews/") else "turntable" if rel.endswith("turntable.png")
                    else "stage" if rel.startswith("evals/eval_") else "image")
            item = {"url": settings.url(p), "kind": kind, "name": p.name, "t": p.stat().st_mtime,
                    "role": info.get("role"), "structureHash": info.get("structure_hash"),
                    "nodeId": info.get("node_id"), "generation": info.get("gen")}
            item.update({k: v for k, v in scores.get(str(p.resolve()), {}).items() if v is not None})
            out.append(item)
    out.sort(key=lambda x: x["t"], reverse=True)
    return out[:limit]


@app.get("/api/runs/{run_id}/agent-events")
def agent_events(run_id: str, structure_hash: str | None = None, node_id: str | None = None, limit: int = 200):
    _require(run_id)
    q: dict[str, Any] = {"scope": run_id}
    if structure_hash:
        q["structure_hash"] = structure_hash
    if node_id:
        q["node_id"] = node_id
    docs = list(db.agent_events.find(q).sort([("ts", -1)]).limit(limit))
    return [_clean(d) for d in reversed(docs)]


@app.get("/api/runs/{run_id}/vectors")
def vectors(run_id: str):
    """Score vectors (8 per structure) and pairwise outcome-vector cosine, for the comparison view."""
    _require(run_id)
    docs = [d for d in db.structures.find({"scope": run_id, "outcome_vec": {"$exists": True}})]
    names = ["pixel", "depth", "normals", "silhouette", "edges", "embedding", "color", "judge"]

    def cos(a: list[float], b: list[float]) -> float:
        na = sum(x * x for x in a) ** 0.5 or 1.0
        nb = sum(x * x for x in b) ** 0.5 or 1.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    return {
        "steps": names,
        "structures": [{"structureHash": d["structure_hash"], "fitness": (d.get("fitness") or {}).get("mean"),
                        "scoreVec": d.get("score_vec"), "roles": [n.get("role") for n in d.get("nodes", [])]}
                       for d in docs],
        "similarity": [[round(cos(a["outcome_vec"], b["outcome_vec"]), 3) for b in docs] for a in docs],
    }


def _latest_model(ctx) -> dict[str, Any] | None:
    """The newest agent's model (exported when each agent finishes), shown until a scored best exists."""
    import json

    glbs = sorted(ctx.dir.glob("nodes/*/node.glb"), key=lambda p: p.stat().st_mtime)
    if not glbs:
        return None
    info: dict[str, Any] = {}
    try:
        info = json.loads((glbs[-1].parent / "node.json").read_text())
    except (OSError, ValueError):
        pass
    return {"url": settings.url(glbs[-1]), "role": info.get("role"), "structureHash": info.get("structure_hash"),
            "generation": info.get("gen"), "t": info.get("t")}


def _require(run_id: str):
    try:
        return manager.get(run_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown run {run_id}") from exc


def _structure_card(d: dict[str, Any]) -> dict[str, Any]:
    f = d.get("fitness") or {}
    return {
        "structureHash": d["structure_hash"], "generation": d.get("generation"), "origin": d.get("origin"),
        "fitness": f.get("mean"), "samples": f.get("n", 0),
        "nodes": [{"nodeId": n.get("node_id"), "role": n.get("role"), "tools": n.get("tools", [])}
                  for n in d.get("nodes", [])],
        "edges": d.get("edges", []), "scores": d.get("scores"), "critiqueFixes": d.get("critique_fixes", []),
        "render": d.get("render"), "glb": d.get("glb"), "overview": d.get("overview"), "costUsd": d.get("cost_usd"),
        "turntable": d.get("turntable"), "frontMatch": d.get("front_match"), "solidity": d.get("solidity"),
        "disqualified": d.get("disqualified") or [],
        "parents": d.get("parent_structure_hashes", []),
    }
