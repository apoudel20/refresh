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

# Atlas Vector Search indexes: {collection: [(index name, vector path, dims, filter fields)]}.
# trait_vec: 64-dim role/tool fingerprint (near-duplicate gate).
# outcome_vec: render-eval's 8x32 token matrix, flattened (what a structure's result looked like).
OUTCOME_DIMS = 256
VECTOR_INDEXES = {
    "agents": [("agent_vec", "trait_vec", VEC_DIMS, ["role"])],
    "structures": [("struct_vec", "trait_vec", VEC_DIMS, ["scope"]),
                   ("outcome_vec", "outcome_vec", OUTCOME_DIMS, ["scope"])],
}


def get_db():
    uri = os.getenv("MONGODB_URI")
    if uri:
        from pymongo import MongoClient
        kwargs = {"serverSelectionTimeoutMS": int(os.getenv("MONGODB_TIMEOUT_MS", "8000"))}
        if uri.startswith("mongodb+srv://") or "tls=true" in uri.lower():
            try:
                import certifi  # standalone Pythons on macOS often lack the system CA bundle
                kwargs["tlsCAFile"] = certifi.where()
            except ImportError:
                pass
        client = MongoClient(uri, **kwargs)
        try:
            client.admin.command("ping")
            return client[os.getenv("LINEAGE_DB", "lineage")]
        except PyMongoError as e:
            if os.getenv("LINEAGE_REQUIRE_ATLAS") == "1":
                raise
            print(f"WARNING: MongoDB at MONGODB_URI is unreachable ({type(e).__name__}). Using in-memory mongomock "
                  "for now: nothing persists. Atlas fix: Network Access -> add this machine's IP address.",
                  flush=True)
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
        for coll, specs in VECTOR_INDEXES.items():
            for name, path, dims, filters in specs:
                if not list(db[coll].list_search_indexes(name)):
                    db[coll].create_search_index(SearchIndexModel(name=name, type="vectorSearch", definition={"fields": [
                        {"type": "vector", "path": path, "numDimensions": dims, "similarity": "cosine"},
                        *({"type": "filter", "path": f} for f in filters)]}))
        deadline = time.time() + wait
        while time.time() < deadline:
            if all(i.get("queryable") for c, specs in VECTOR_INDEXES.items() for (n, *_rest) in specs
                   for i in db[c].list_search_indexes(n)):
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


def similar_outcomes(db, scope, vec, k=3, exclude=None):
    """Structures in this scope whose render-eval outcome vectors are closest to ``vec``.

    Atlas: $vectorSearch on the ``outcome_vec`` index. mongomock / no index: exact cosine in Python.
    Returns [{structure_hash, fitness, nodes, critique_fixes, similarity}] (similarity = cosine).
    """
    proj = {"_id": 0, "structure_hash": 1, "fitness": 1, "nodes": 1, "critique_fixes": 1, "generation": 1}
    if isinstance(db, Database):
        try:
            out = []
            for d in db.structures.aggregate([
                    {"$vectorSearch": {"index": "outcome_vec", "path": "outcome_vec", "queryVector": list(vec),
                                       "numCandidates": 60, "limit": k + 1, "filter": {"scope": scope}}},
                    {"$project": {**proj, "score": {"$meta": "vectorSearchScore"}}}]):
                if d["structure_hash"] != exclude:
                    d["similarity"] = 2 * d.pop("score") - 1  # Atlas reports cosine as (1 + cos) / 2
                    out.append(d)
            return out[:k]
        except PyMongoError:
            pass
    q = [float(x) for x in vec]
    qn = sum(x * x for x in q) ** 0.5 or 1.0
    scored = []
    for d in db.structures.find({"scope": scope, "outcome_vec": {"$exists": True}}, {**proj, "outcome_vec": 1}):
        if d.get("structure_hash") == exclude:
            continue
        v = d.pop("outcome_vec") or []
        vn = sum(x * x for x in v) ** 0.5 or 1.0
        d["similarity"] = sum(a * b for a, b in zip(q, v)) / (qn * vn)
        scored.append(d)
    return sorted(scored, key=lambda d: -d["similarity"])[:k]
