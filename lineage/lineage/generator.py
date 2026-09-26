"""Proposes structures: LLM-guided from memory, plus cheap genetic operators on elites (spec §4)."""
import random

from . import llm
from .hashing import MAX_DEPTH, MAX_NODES, agent_hash, normalize_genome, trait_vec, validate
from .store import similar_outcomes

ROLES = ["planner", "preprocessor", "geometry", "refiner", "texturer", "verifier"]

GEN_SYSTEM = ("You design small pipelines of AI agents (a DAG with one root and at most 6 nodes; the longest path "
              "has at most 3 agents, e.g. n0 -> n1 -> n2, so a 4-agent chain is invalid; edges hand work from one "
              "agent to the next, and a node may have several parents or children). Use the roles " + ", ".join(ROLES)
              + ". Each agent has a role, a one-line brief and a set of "
              "allowed tools. Learn from the memory you're given: build on high scorers, avoid what already failed, "
              "and never repeat a structure listed as tried. Reply with JSON: "
              '{"structures": [{"nodes": [{"id": "n0", "role": "...", "brief": "...", "tools": ["..."]}], '
              '"edges": [{"from": "n0", "to": "n1", "type": "then"}]}]}')


class Generator:
    def __init__(self, db, scope, registry, use_memory, seed=0, model=None, task_brief=""):
        self.db, self.scope, self.use_memory, self.model = db, scope, use_memory, model
        self.tools = [t["tool_id"] for t in registry]
        self.tool_desc = {t["tool_id"]: t.get("description", "") for t in registry}
        self.task_brief = task_brief
        self.rng = random.Random(seed)
        self.last_gen = []  # memory-off keeps only the previous generation, in process

    def genome(self, role, tools, brief=None, created_by="generator", parents=()):
        g = normalize_genome({"role": role, "brief": brief or f"Act as the {role} for the task.",
                              "model": self.model, "tools": [t for t in tools if t in self.tools],
                              "delegates": [], "params": {"temperature": 0.3, "max_calls": 6}})
        g["agent_hash"] = agent_hash(g)
        self.db.agents.update_one({"agent_hash": g["agent_hash"]}, {"$setOnInsert": {
            **g, "created_by": created_by, "parent_agent_hashes": list(parents)},
            "$set": {"trait_vec": trait_vec(g["role"], g["tools"])}}, upsert=True)
        return g

    def _random_tools(self):
        return self.rng.sample(self.tools, self.rng.randint(1, 3))

    def random_structure(self):
        n = self.rng.randint(1, 4)
        nodes = {f"n{i}": self.genome(self.rng.choice(ROLES), self._random_tools()) for i in range(n)}
        edges = [{"from": f"n{i - 1}", "to": f"n{i}", "type": "then", "order": i} for i in range(1, n)]
        return {"nodes": nodes, "edges": edges, "origin": "random", "parents": []}

    def mutate(self, parent):
        nodes, edges = dict(parent["nodes"]), [dict(e) for e in parent["edges"]]
        op = self.rng.choice(["swap_tools", "add_node", "remove_leaf", "swap_tools"])
        leaves = [n for n in nodes if not any(e["from"] == n for e in edges)]
        if op == "add_node" and len(nodes) < MAX_NODES:
            nid = f"n{max(int(k[1:]) for k in nodes) + 1}"
            src = self.rng.choice(leaves)
            nodes[nid] = self.genome(self.rng.choice(ROLES), self._random_tools(), created_by="mutation")
            edges.append({"from": src, "to": nid, "type": "then", "order": len(edges) + 1})
        elif op == "remove_leaf" and len(nodes) > 1:
            gone = self.rng.choice(leaves)
            nodes.pop(gone)
            edges = [e for e in edges if gone not in (e["from"], e["to"])]
        else:
            nid = self.rng.choice(list(nodes))
            old = nodes[nid]
            tools = set(old["tools"])
            tools.symmetric_difference_update({self.rng.choice(self.tools)})
            nodes[nid] = self.genome(old["role"], sorted(tools) or self._random_tools(), old["brief"],
                                     created_by="mutation", parents=[old["agent_hash"]])
            op = "swap_tools"
        return {"nodes": nodes, "edges": edges, "origin": f"mutation:{op}", "parents": [parent["structure_hash"]]}

    def elites(self, k=3):
        if not self.use_memory:
            return sorted(self.last_gen, key=lambda s: -s["fitness"])[:k]
        docs = self.db.structures.find({"scope": self.scope, "fitness.n": {"$gte": 1}}).sort("fitness.mean", -1).limit(k)
        out = []
        for d in docs:
            genomes = {a["agent_hash"]: a for a in self.db.agents.find({"agent_hash": {"$in": [n["agent_hash"] for n in d["nodes"]]}})}
            out.append({"structure_hash": d["structure_hash"], "fitness": d["fitness"]["mean"], "edges": d["edges"],
                        "nodes": {n["node_id"]: {k: v for k, v in genomes[n["agent_hash"]].items() if k != "_id"} for n in d["nodes"]}})
        return out

    def _describe(self, s):
        nodes = "; ".join(f"{nid}={g['role']}[{','.join(g['tools'])}]" for nid, g in s["nodes"].items())
        edges = ", ".join(f"{e['from']}->{e['to']}" for e in s["edges"]) or "none"
        return f"nodes: {nodes} | edges: {edges}"

    def llm_proposals(self, n):
        mem = ""
        if self.use_memory:
            top = self.elites(5)
            low = list(self.db.structures.find({"scope": self.scope, "fitness.n": {"$gte": 1}}).sort("fitness.mean", 1).limit(5))
            tried = self.db.structures.count_documents({"scope": self.scope})
            mem = ("Best so far:\n" + "\n".join(f"- fitness {s['fitness']:.2f}: {self._describe(s)}" for s in top) +
                   "\nWorst so far (avoid):\n" + "\n".join(f"- fitness {d['fitness']['mean']:.2f}: roles "
                   + ", ".join(n['role'] for n in d['nodes']) for d in low) + f"\nStructures already tried: {tried}.")
            mem += self._outcome_memory(top[0]["structure_hash"] if top else None)
        tools_txt = "\n".join(f"- {t}: {self.tool_desc.get(t, '')}" for t in self.tools)
        task = f"Task: {self.task_brief}\n" if self.task_brief else ""
        out = llm.ask_json(GEN_SYSTEM, f"{task}Available tools:\n{tools_txt}\n{mem}\nPropose {n} new structures.",
                           model=self.model, temperature=0.8)
        props = []
        for s in (out or {}).get("structures", [])[:n]:
            try:
                nodes = {x["id"]: self.genome(x.get("role", "agent"), x.get("tools", []), x.get("brief")) for x in s["nodes"]}
                edges = [{"from": e["from"], "to": e["to"], "type": e.get("type", "then"), "order": i}
                         for i, e in enumerate(s.get("edges", []))]
                nodes, edges = _trim(nodes, edges)
                validate({k: g["agent_hash"] for k, g in nodes.items()}, edges)
                props.append({"nodes": nodes, "edges": edges, "origin": "llm", "parents": []})
            except (KeyError, ValueError, TypeError) as e:
                print(f"WARNING: dropped an invalid LLM structure ({type(e).__name__}: {e})", flush=True)
                continue
        return props

    def _outcome_memory(self, best_hash):
        """What the critic said about the best structure, and which structures failed the same way."""
        if not best_hash:
            return ""
        doc = self.db.structures.find_one({"scope": self.scope, "structure_hash": best_hash},
                                          {"critique_fixes": 1, "outcome_vec": 1, "scores": 1})
        if not doc:
            return ""
        parts = []
        if doc.get("scores"):
            weak = sorted(doc["scores"].items(), key=lambda kv: kv[1])[:3]
            parts.append("Weakest render-eval steps of the best structure: "
                         + ", ".join(f"{k} {v:.2f}" for k, v in weak))
        if doc.get("critique_fixes"):
            parts.append("Critic's top fixes for the best structure: " + " | ".join(doc["critique_fixes"][:3]))
        if doc.get("outcome_vec"):
            sims = similar_outcomes(self.db, self.scope, doc["outcome_vec"], k=3, exclude=best_hash)
            if sims:
                parts.append("Structures whose results looked most like the best one (same strengths and failures): "
                             + "; ".join(f"fitness {(d.get('fitness') or {}).get('mean', 0):.2f}, roles "
                                         + ",".join(n["role"] for n in d.get("nodes", [])) for d in sims))
        return ("\n" + "\n".join(parts)) if parts else ""

    def propose(self, k):
        props = self.llm_proposals(k // 2) if llm.enabled() else []
        elites = self.elites()
        while len(props) < k:
            # exploit: mostly mutate elites, biased to the best; explore: the odd random structure
            props.append(self.mutate(elites[min(int(self.rng.expovariate(1.2)), len(elites) - 1)])
                         if elites and self.rng.random() < 0.85 else self.random_structure())
        return props


def _trim(nodes, edges):
    """Salvage an over-deep or over-large LLM proposal: keep nodes within MAX_DEPTH levels (and MAX_NODES)."""
    preds = {n: [e["from"] for e in edges if e["to"] == n and e["from"] in nodes] for n in nodes}
    depth = {}

    def d(n, seen=()):
        if n in seen:
            raise ValueError("cycle")
        if n not in depth:
            depth[n] = 1 + max((d(p, seen + (n,)) for p in preds[n]), default=0)
        return depth[n]

    keep = sorted((n for n in nodes if d(n) <= MAX_DEPTH), key=lambda n: depth[n])[:MAX_NODES]
    if len(keep) == len(nodes):
        return nodes, edges
    kept = set(keep)
    edges = [dict(e, order=i) for i, e in enumerate(e for e in edges if e["from"] in kept and e["to"] in kept)]
    return {n: nodes[n] for n in keep}, edges
