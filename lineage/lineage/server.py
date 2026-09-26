"""Live orchestration view: change stream on `events` → server-sent events → ui/index.html (spec §10)."""
import asyncio
import json
import pathlib
import threading
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pymongo.errors import PyMongoError

from .store import ensure_indexes, get_db

app = FastAPI()
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
