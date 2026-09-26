"""Atlas access. Falls back to mongomock when MONGODB_URI is unset (local dev only; finalists must use Atlas)."""
import os
import time

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING

load_dotenv()


def get_db():
    uri = os.getenv("MONGODB_URI")
    if uri:
        from pymongo import MongoClient
        return MongoClient(uri)[os.getenv("LINEAGE_DB", "lineage")]
    import mongomock
    global _MOCK
    _MOCK = globals().get("_MOCK") or mongomock.MongoClient()
    return _MOCK["lineage"]


def ensure_indexes(db):
    db.agents.create_index([("agent_hash", ASCENDING)], unique=True)
    db.structures.create_index([("scope", ASCENDING), ("structure_hash", ASCENDING)], unique=True)
    db.structures.create_index([("scope", ASCENDING), ("fitness.mean", DESCENDING)])
    db.modules.create_index([("scope", ASCENDING), ("trace_key", ASCENDING)], unique=True)
    db.modules.create_index([("scope", ASCENDING), ("input_key", ASCENDING)])
    db.events.create_index([("scope", ASCENDING), ("t", ASCENDING)])


def log(db, scope, kind, **detail):
    db.events.insert_one({"t": time.time(), "scope": scope, "kind": kind, **detail})


def record_fitness(db, scope, s_hash, fitness):
    """Welford update of {mean, n, var} so repeated samples of one structure average out."""
    doc = db.structures.find_one({"scope": scope, "structure_hash": s_hash}, {"fitness": 1})
    f = (doc or {}).get("fitness") or {"mean": 0.0, "n": 0, "m2": 0.0}
    n = f["n"] + 1
    delta = fitness - f["mean"]
    mean = f["mean"] + delta / n
    m2 = f["m2"] + delta * (fitness - mean)
    new = {"mean": mean, "n": n, "m2": m2, "var": m2 / (n - 1) if n > 1 else 0.0}
    db.structures.update_one({"scope": scope, "structure_hash": s_hash}, {"$set": {"fitness": new}})
    return new
