"""Command line: `refresh-server` (API for uipack) and `refresh-search` (headless run on one image)."""

from __future__ import annotations

import argparse
import time


def serve() -> None:
    p = argparse.ArgumentParser(prog="refresh-server", description="Refresh API (uipack backend)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    a = p.parse_args()
    import uvicorn

    uvicorn.run("refresh.api:app", host=a.host, port=a.port, reload=a.reload)


def headless() -> None:
    p = argparse.ArgumentParser(prog="refresh-search", description="Run one reconstruction search without the UI")
    p.add_argument("reference", help="reference image")
    p.add_argument("extra_views", nargs="*", help="extra reference views (context only)")
    p.add_argument("--project", default="cli")
    a = p.parse_args()
    from lineage.store import get_db

    from .runs import RunManager

    db = get_db()
    manager = RunManager(db)
    images = [(path, open(path, "rb").read()) for path in [a.reference, *a.extra_views]]
    ctx = manager.create(a.project, images)
    print(f"run {ctx.run_id} -> {ctx.dir}")
    manager.start(ctx)
    seen = 0.0
    while manager.threads[ctx.run_id].is_alive():
        for e in db.events.find({"scope": ctx.scope, "t": {"$gt": seen}}).sort("t", 1):
            seen = e["t"]
            extra = {k: e[k] for k in ("gen", "node_id", "fitness", "reason", "best") if k in e}
            print(f"{e['kind']:<20} {extra}")
        time.sleep(1.0)
    best = db.structures.find_one({"scope": ctx.scope, "fitness.n": {"$gte": 1}}, sort=[("fitness.mean", -1)])
    if best:
        print(f"best fitness {best['fitness']['mean']:.3f}: {best.get('glb')}")
        print(best.get("critique", ""))
