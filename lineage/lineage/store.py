"""Atlas access. Falls back to mongomock when MONGODB_URI is unset (local dev only; finalists must use Atlas)."""
import os
import time

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

from .hashing import VEC_DIMS

load_dotenv()

# Atlas Vector Search indexes over trait vectors: {collection: (index name, filter fields)}.
VECTOR_INDEXES = {"agents": ("agent_vec", ["role"]), "structures": ("struct_vec", ["scope"])}


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


def ensure_vector_indexes(db, wait=180):
    """Atlas only: create the trait-vector indexes if missing and wait until queryable. False on mongomock or failure."""
    if not isinstance(db, Database):
        return False
    try:
        for coll, (name, filters) in VECTOR_INDEXES.items():
            if not list(db[coll].list_search_indexes(name)):
                db[coll].create_search_index(SearchIndexModel(name=name, type="vectorSearch", definition={"fields": [
                    {"type": "vector", "path": "trait_vec", "numDimensions": VEC_DIMS, "similarity": "cosine"},
                    *({"type": "filter", "path": f} for f in filters)]}))
        deadline = time.time() + wait
        while time.time() < deadline:
            if all(i.get("queryable") for c, (n, _) in VECTOR_INDEXES.items() for i in db[c].list_search_indexes(n)):
                return True
            time.sleep(3)
        print("vector indexes not queryable yet; near-duplicate gate off for this run")
    except PyMongoError as e:
        print(f"vector indexes unavailable ({e}); near-duplicate gate off")
    return False


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
