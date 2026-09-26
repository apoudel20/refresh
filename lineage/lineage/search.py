"""The search loop: propose → gate → run → evaluate → remember (spec §3). Resumable: rerun with the same --scope."""
import argparse
import pathlib
import time

from pymongo.errors import PyMongoError

from .generator import Generator
from .hashing import H, content_hash, namespace, structure_hash, structure_vec
from .runner import Runner
from .store import ensure_indexes, ensure_vector_indexes, get_db, log, record_fitness
from .workbench import make_workbench


def gate(db, scope, s_hash, elite_hashes):
    """Memory-on only. Returns None to allow, or a reason to block."""
    doc = db.structures.find_one({"scope": scope, "structure_hash": s_hash}, {"fitness": 1, "generation": 1})
    if not doc or not doc.get("fitness"):
        return None
    if s_hash in elite_hashes and doc["fitness"]["n"] < 3:
        return None  # deliberate re-sample of an elite
    return f"tried in gen {doc.get('generation')}"


NEAR_DUP = 0.97  # cosine similarity of mean trait vectors


def near_dup(db, scope, s_hash, vec):
    """Atlas $vectorSearch over evaluated structures in this scope. Returns the closest one if cosine >= NEAR_DUP."""
    try:
        for d in db.structures.aggregate([
                {"$vectorSearch": {"index": "struct_vec", "path": "trait_vec", "queryVector": vec,
                                   "numCandidates": 50, "limit": 3, "filter": {"scope": scope}}},
                {"$project": {"structure_hash": 1, "generation": 1, "score": {"$meta": "vectorSearchScore"}}}]):
            if d["structure_hash"] != s_hash:
                d["cosine"] = 2 * d["score"] - 1  # Atlas reports cosine as (1 + cos) / 2
                return d if d["cosine"] >= NEAR_DUP else None
    except PyMongoError as e:
        print(f"$vectorSearch failed ({e}); allowing")
    return None


# Structure-level fields a rich evaluator (Refresh: render-eval) may return, stored on the structure doc.
EVAL_EXTRAS = ("critique", "critique_fixes", "scores", "score_vec", "outcome_vec", "eval_dir", "render", "glb",
               "overview", "blend", "front_match", "solidity", "disqualified", "turntable")


def search(scope, memory, generations, k, workbench, task_id, task_input, seed=0, model=None, db=None,
           stop_event=None):
    db = db if db is not None else get_db()
    ensure_indexes(db)
    vec_gate = memory and ensure_vector_indexes(db)
    registry = workbench.tools()
    if hasattr(workbench, "task_artifact"):  # the workbench knows what the task input really is
        task_art = workbench.task_artifact(task_id, task_input)
        task_hash = task_art["hash"]
    else:
        task_hash = content_hash(pathlib.Path(task_input).read_bytes() if pathlib.Path(task_input).is_file() else task_input)
        task_art = {"ref": f"task:{task_hash[:12]}", "hash": task_hash, "chain": [], "summary": f"task input {task_id}"}
    ns = namespace(task_id, task_hash, workbench.eval_version, H(registry))
    gen0 = 1 + max([d.get("generation", -1) for d in db.structures.find({"scope": scope}, {"generation": 1})], default=-1)
    gen_ = Generator(db, scope, registry, memory, seed=seed + gen0, model=model,
                     task_brief=getattr(workbench, "task_brief", ""))
    runner = Runner(db, workbench, scope, ns, registry, use_cache=memory)
    log(db, scope, "resume" if gen0 else "search_started", memory=memory, gen=gen0, vector_gate=vec_gate)
    for gen in range(gen0, gen0 + generations):
        if stop_event is not None and stop_event.is_set():
            log(db, scope, "stopped", gen=gen)
            break
        elite_hashes = {e["structure_hash"] for e in gen_.elites()} if memory else set()
        this_gen = []
        for cand in gen_.propose(k):
            if stop_event is not None and stop_event.is_set():
                break
            nodes = cand["nodes"]
            try:
                s_hash = structure_hash(ns, {n: g["agent_hash"] for n, g in nodes.items()}, cand["edges"])
            except ValueError as e:
                log(db, scope, "invalid", gen=gen, reason=str(e), origin=cand["origin"])
                continue
            existing = db.structures.find_one({"scope": scope, "structure_hash": s_hash}, {"generation": 1})
            node_docs = [{"node_id": n, "agent_hash": g["agent_hash"], "role": g["role"], "tools": g["tools"]}
                         for n, g in nodes.items()]
            if memory:
                reason = gate(db, scope, s_hash, elite_hashes)
                if reason:
                    log(db, scope, "blocked", gen=gen, structure_hash=s_hash, reason=reason, origin=cand["origin"],
                        nodes=node_docs, edges=cand["edges"])
                    continue
                dup = vec_gate and not existing and near_dup(db, scope, s_hash, structure_vec(node_docs))
                if dup:
                    log(db, scope, "near_dup", gen=gen, structure_hash=s_hash, similar_to=dup["structure_hash"],
                        score=round(dup["cosine"], 3), reason=f"~{dup['structure_hash'][:8]} gen {dup.get('generation')}",
                        origin=cand["origin"], nodes=node_docs, edges=cand["edges"])
                    continue
            elif existing:
                log(db, scope, "repeat", gen=gen, structure_hash=s_hash, first_gen=existing.get("generation"))
            db.structures.update_one({"scope": scope, "structure_hash": s_hash}, {"$setOnInsert": {
                "scope": scope, "structure_hash": s_hash, "ns": ns, "generation": gen, "nodes": node_docs,
                "edges": cand["edges"], "origin": cand["origin"], "parent_structure_hashes": cand["parents"],
                "created": time.time()}}, upsert=True)
            log(db, scope, "structure_started", gen=gen, structure_hash=s_hash, origin=cand["origin"],
                nodes=node_docs, edges=cand["edges"])
            try:
                sinks, cost = runner.run(s_hash, nodes, cand["edges"], task_art, gen)
                ev = workbench.evaluate(task_id, s_hash, sinks)
            except ConnectionError as e:  # Blender (or another hard dependency) is gone: stop cleanly
                log(db, scope, "search_failed", gen=gen, structure_hash=s_hash, reason=str(e))
                return db
            except Exception as e:  # one broken structure must not end a long search
                log(db, scope, "structure_failed", gen=gen, structure_hash=s_hash, reason=f"{type(e).__name__}: {e}")
                continue
            fit = record_fitness(db, scope, s_hash, ev["fitness"])
            extras = {x: ev[x] for x in EVAL_EXTRAS if ev.get(x) is not None}
            db.structures.update_one({"scope": scope, "structure_hash": s_hash},
                                     {"$set": {"metrics": ev.get("metrics"), "per_node": ev.get("per_node"), "cost_usd": cost,
                                              "trait_vec": structure_vec(node_docs), **extras}})
            log(db, scope, "eval", gen=gen, structure_hash=s_hash, fitness=ev["fitness"], mean=fit["mean"], n=fit["n"],
                cost_usd=cost, scores=ev.get("scores"), glb=ev.get("glb"), render=ev.get("render"))
            this_gen.append({**cand, "structure_hash": s_hash, "fitness": ev["fitness"]})
        gen_.last_gen = this_gen or gen_.last_gen
        best = db.structures.find_one({"scope": scope, "fitness.n": {"$gte": 1}}, sort=[("fitness.mean", -1)])
        log(db, scope, "generation_done", gen=gen, best=best["fitness"]["mean"] if best else None)
    best = db.structures.find_one({"scope": scope, "fitness.n": {"$gte": 1}}, sort=[("fitness.mean", -1)])
    log(db, scope, "search_done", best=best["fitness"]["mean"] if best else None,
        best_structure=best["structure_hash"] if best else None, glb=(best or {}).get("glb"))
    return db


def main():
    p = argparse.ArgumentParser(description="Lineage: memory-backed search over agent structures")
    p.add_argument("--scope", required=True, help="memory namespace; rerun the same scope to resume")
    p.add_argument("--memory", choices=["on", "off"], default="on")
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--k", type=int, default=4, help="structures proposed per generation")
    p.add_argument("--workbench", default="mock", help="'mock' or the workbench base URL (tools)")
    p.add_argument("--eval-url", default=None, help="eval engine base URL, if separate from the workbench")
    p.add_argument("--task-id", default="recon-demo")
    p.add_argument("--task-input", default="demo-image", help="path to the task input file, or a literal id")
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    wb = make_workbench(a.workbench, a.eval_url, a.seed)
    search(a.scope, a.memory == "on", a.generations, a.k, wb, a.task_id, a.task_input, a.seed, a.model)


if __name__ == "__main__":
    main()
