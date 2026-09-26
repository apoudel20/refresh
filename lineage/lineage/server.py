"""Live orchestration view: change stream on `events` → server-sent events → ui/index.html (spec §10)."""
import asyncio
import json
import pathlib
import threading
import os
import time
import uuid

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pymongo.errors import PyMongoError

from .store import ensure_indexes, get_db

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
db = get_db()
ensure_indexes(db)
UI = pathlib.Path(__file__).resolve().parent.parent / "ui" / "index.html"
STOPS = {}  # scope -> threading.Event for searches started here


def _clean(d):
    d = dict(d)
    d.pop("_id", None)
    return d


@app.get("/")
def index():
    return FileResponse(UI)


@app.get("/how")
def how():
    return FileResponse(UI.parent / "how.html")


@app.post("/api/stop")
def stop(scope: str):
    if scope in STOPS:
        STOPS[scope].set()
    return {"stopping": scope in STOPS}


@app.get("/api/scopes")
def scopes():
    return sorted(db.structures.distinct("scope"))


@app.get("/api/state")
def state(scope: str):
    events = [_clean(e) for e in db.events.find({"scope": scope}).sort("t", 1)]
    return {"events": events}


@app.post("/api/run")
def run(scope: str, memory: str = "on", generations: int = 4, k: int = 4, workbench: str = "mock", eval_url: str = "",
        task_id: str = "recon-demo", task_input: str = "demo-image", seed: int = 0):
    """Start a search in this process (needed for local mongomock; works the same on Atlas)."""
    from .search import search
    from .workbench import make_workbench
    wb = make_workbench(workbench, eval_url or None, seed)
    STOPS[scope] = threading.Event()
    threading.Thread(target=search, daemon=True, kwargs=dict(
        scope=scope, memory=memory == "on", generations=generations, k=k, workbench=wb,
        task_id=task_id, task_input=task_input, seed=seed, db=db, stop=STOPS[scope])).start()
    return {"started": scope}


@app.get("/api/module")
def module(scope: str, trace_key: str):
    m = db.modules.find_one({"scope": scope, "trace_key": trace_key})
    return _clean(m) if m else {}


@app.get("/api/lineage")
def lineage(scope: str, structure_hash: str):
    """Family tree of a structure via $graphLookup over parent_structure_hashes (Atlas only)."""
    try:
        doc = next(db.structures.aggregate([
            {"$match": {"scope": scope, "structure_hash": structure_hash}},
            {"$graphLookup": {"from": "structures", "startWith": "$parent_structure_hashes",
                              "connectFromField": "parent_structure_hashes", "connectToField": "structure_hash",
                              "as": "ancestors", "restrictSearchWithMatch": {"scope": scope}}},
            {"$project": {"_id": 0, "ancestors.structure_hash": 1, "ancestors.generation": 1,
                          "ancestors.fitness": 1, "ancestors.origin": 1}}]), {})
        return doc
    except (PyMongoError, NotImplementedError):
        return {"ancestors": []}


def _watch(scope, queue, loop, stop, since=0.0):
    """Replay anything after `since`, then prefer a real change stream (Atlas); fall back to polling (mongomock)."""
    last = since
    for e in db.events.find({"scope": scope, "t": {"$gt": since}}).sort("t", 1):
        last = e["t"]
        loop.call_soon_threadsafe(queue.put_nowait, _clean(e))
    try:
        with db.events.watch([{"$match": {"operationType": "insert", "fullDocument.scope": scope}}]) as cs:
            for ch in cs:
                if stop.is_set():
                    return
                loop.call_soon_threadsafe(queue.put_nowait, _clean(ch["fullDocument"]))
    except (PyMongoError, NotImplementedError, AttributeError, TypeError):
        while not stop.is_set():
            for e in db.events.find({"scope": scope, "t": {"$gt": last}}).sort("t", 1):
                last = e["t"]
                loop.call_soon_threadsafe(queue.put_nowait, _clean(e))
            time.sleep(0.4)


@app.get("/api/stream")
async def stream(scope: str, since: float = 0.0):
    queue, stop = asyncio.Queue(), threading.Event()
    threading.Thread(target=_watch, args=(scope, queue, asyncio.get_running_loop(), stop, since), daemon=True).start()

    async def gen():
        try:
            while True:
                try:
                    e = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(e, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            stop.set()

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Refresh UI contract (uipack/app/lib/harness-client.ts): an upload starts a Lineage search on that image ──
RUNS = pathlib.Path(os.getenv("LINEAGE_WORKDIR", "lineage_work")).resolve() / "runs"
WORKBENCH = os.getenv("LINEAGE_WORKBENCH", "mock")  # or the Blender workbench URL, e.g. http://localhost:8150
DEMO_MODEL = os.getenv("LINEAGE_DEMO_MODEL", "http://localhost:3000/assets/demo-pigeon.glb")
RECON = {}  # run_id -> {"thread", "image", "model"}


def _start_recon(run_id, image, generations):
    from .search import search
    from .workbench import make_workbench
    t = threading.Thread(target=search, daemon=True, kwargs=dict(
        scope=run_id, memory=True, generations=generations, k=int(os.getenv("RECON_K", "4")),
        workbench=make_workbench(WORKBENCH), task_id="refresh-ui", task_input=image, db=db))
    RECON.setdefault(run_id, {"image": image})["thread"] = t
    t.start()


@app.post("/api/reconstructions")
async def reconstruct(images: list[UploadFile] = File(...), projectId: str = Form("")):
    run_id = f"recon-{uuid.uuid4().hex[:8]}"
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    image = d / f"reference{pathlib.Path(images[0].filename or 'x.png').suffix or '.png'}"
    image.write_bytes(await images[0].read())
    _start_recon(run_id, str(image), int(os.getenv("RECON_GENERATIONS", "3")))
    return {"runId": run_id, "eventsUrl": f"/api/reconstructions/{run_id}/events"}


def _update(e):
    k, g = e["kind"], e.get("gen")
    if k in ("search_started", "resume"):
        return {"type": "status", "status": f"Lineage: searching agent structures from memory (generation {g})"}
    if k == "structure_started":
        return {"type": "status", "status": f"Generation {g}: running a {len(e.get('nodes', []))}-agent structure ({e.get('origin')})"}
    if k == "blocked":
        return {"type": "status", "status": "Memory: skipped a structure already tried"}
    if k == "near_dup":
        return {"type": "status", "status": f"Memory: skipped a near-duplicate ($vectorSearch {e.get('score', 0):.2f})"}
    if k == "cache_hit":
        return {"type": "status", "status": "Memory: reused a node's output"}
    if k == "generation_done" and e.get("best") is not None:
        return {"type": "metric", "similarity": round(100 * e["best"], 1), "status": f"Generation {g} done"}
    return None


def _model_url(run_id, base):
    """GLB of the best structure: exported by the Blender workbench, or the UI's demo model on the mock workbench."""
    best = db.structures.find_one({"scope": run_id, "fitness.n": {"$gte": 1}}, sort=[("fitness.mean", -1)])
    ref = ((best or {}).get("outputs") or [{}])[0].get("ref")
    if WORKBENCH != "mock" and ref and os.path.isfile(ref):
        import requests
        RECON[run_id]["model"] = requests.post(f"{WORKBENCH.rstrip('/')}/export", json={"ref": ref}, timeout=300).json()["path"]
        return f"{base}api/reconstructions/{run_id}/model.glb?v={uuid.uuid4().hex[:6]}", (best or {}).get("structure_hash")
    return DEMO_MODEL, (best or {}).get("structure_hash")


@app.get("/api/reconstructions/{run_id}/events")
async def recon_events(run_id: str, request: Request):
    base = str(request.base_url)

    async def gen():
        last = 0.0
        while True:
            evs = await asyncio.to_thread(lambda: list(db.events.find({"scope": run_id, "t": {"$gt": last}}).sort("t", 1)))
            for e in evs:
                last = e["t"]
                u = _update(e)
                if u:
                    yield f"data: {json.dumps(u)}\n\n"
            t = RECON.get(run_id, {}).get("thread")
            if not evs and (t is None or not t.is_alive()):
                url, mid = await asyncio.to_thread(_model_url, run_id, base)
                yield f"data: {json.dumps({'type': 'model', 'modelUrl': url, 'modelId': mid, 'status': 'Model ready'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/reconstructions/{run_id}/model.glb")
def recon_model(run_id: str):
    return FileResponse(RECON[run_id]["model"], media_type="model/gltf-binary")


@app.post("/api/reconstructions/{run_id}/refine")
async def recon_refine(run_id: str, request: Request):
    """Refinement = resume the same search for one more generation, building on its memory."""
    _start_recon(run_id, RECON[run_id]["image"], 1)
    await asyncio.to_thread(RECON[run_id]["thread"].join)
    url, mid = await asyncio.to_thread(_model_url, run_id, str(request.base_url))
    return {"modelUrl": url, "modelId": mid}
