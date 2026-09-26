"""Live orchestration view: change stream on `events` -> server-sent events -> ui/index.html (spec §10).

The endpoints live on ``router`` so another app (refresh-server) can include them; ``app`` is the
standalone server. The database is created lazily (``set_db`` lets the host app share its own).
"""
import asyncio
import json
import pathlib
import threading
import time

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pymongo.errors import PyMongoError

from .store import ensure_indexes, get_db

UI = pathlib.Path(__file__).resolve().parent.parent / "ui" / "index.html"
router = APIRouter()
_DB = None


def db():
    global _DB
    if _DB is None:
        _DB = get_db()
        ensure_indexes(_DB)
    return _DB


def set_db(database):
    global _DB
    _DB = database
    ensure_indexes(database)


def _clean(d):
    d = dict(d)
    d.pop("_id", None)
    return d


@router.get("/lineage-ui")
def lineage_ui():
    return FileResponse(UI)


@router.get("/api/scopes")
def scopes():
    return sorted(db().structures.distinct("scope"))


@router.get("/api/state")
def state(scope: str):
    events = [_clean(e) for e in db().events.find({"scope": scope}).sort("t", 1)]
    return {"events": events}


@router.post("/api/run")
def run(scope: str, memory: str = "on", generations: int = 4, k: int = 4, workbench: str = "mock", eval_url: str = "",
        task_id: str = "recon-demo", task_input: str = "demo-image", seed: int = 0):
    """Start a mock/HTTP-workbench search in this process (Blender runs start from refresh-server)."""
    from .search import search
    from .workbench import make_workbench
    wb = make_workbench(workbench, eval_url or None, seed)
    threading.Thread(target=search, daemon=True, kwargs=dict(
        scope=scope, memory=memory == "on", generations=generations, k=k, workbench=wb,
        task_id=task_id, task_input=task_input, seed=seed, db=db())).start()
    return {"started": scope}


@router.get("/api/module")
def module(scope: str, trace_key: str):
    m = db().modules.find_one({"scope": scope, "trace_key": trace_key})
    return _clean(m) if m else {}


@router.get("/api/lineage")
def lineage(scope: str, structure_hash: str):
    """Family tree of a structure via $graphLookup over parent_structure_hashes (Atlas only)."""
    try:
        doc = next(db().structures.aggregate([
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
    database = db()
    last = since
    for e in database.events.find({"scope": scope, "t": {"$gt": since}}).sort("t", 1):
        last = e["t"]
        loop.call_soon_threadsafe(queue.put_nowait, _clean(e))
    try:
        with database.events.watch([{"$match": {"operationType": "insert", "fullDocument.scope": scope}}]) as cs:
            for ch in cs:
                if stop.is_set():
                    return
                loop.call_soon_threadsafe(queue.put_nowait, _clean(ch["fullDocument"]))
    except (PyMongoError, NotImplementedError, AttributeError, TypeError):
        while not stop.is_set():
            for e in database.events.find({"scope": scope, "t": {"$gt": last}}).sort("t", 1):
                last = e["t"]
                loop.call_soon_threadsafe(queue.put_nowait, _clean(e))
            time.sleep(0.4)


def event_stream(scope: str, since: float = 0.0, transform=None):
    """Async SSE generator over a scope's events. ``transform(event) -> list[dict | str]`` may rewrite them."""
    async def gen():
        queue, stop = asyncio.Queue(), threading.Event()
        threading.Thread(target=_watch, args=(scope, queue, asyncio.get_running_loop(), stop, since), daemon=True).start()
        try:
            while True:
                try:
                    e = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                for out in (transform(e) if transform else [e]):
                    yield f"data: {out if isinstance(out, str) else json.dumps(out, default=str)}\n\n"
                    if out == "[DONE]":
                        return
        finally:
            stop.set()
    return gen()


@router.get("/api/stream")
async def stream(scope: str, since: float = 0.0):
    return StreamingResponse(event_stream(scope, since), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app = FastAPI()
app.include_router(router)


@app.get("/")
def index():
    return FileResponse(UI)
