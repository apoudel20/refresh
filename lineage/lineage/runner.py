"""Executes one structure's DAG. Nodes are stateless processors; every run is written to memory (spec §3, §7)."""
import time

from . import llm
from .hashing import input_key, tool_sig, trace_key
from .store import log

AGENT_SYSTEM = ("You are one agent in a pipeline. Pick the tool calls to make, in order. You may repeat or "
                "interleave tools. Only use your allowed tools. Reply with JSON: "
                '{"calls": [{"tool_id": "...", "args": {}}], "note": "one line on why"}')


def topo(nodes, edges):
    preds = {n: [e["from"] for e in edges if e["to"] == n] for n in nodes}
    order, done = [], set()
    while len(order) < len(nodes):
        for n in sorted(nodes):
            if n not in done and all(p in done for p in preds[n]):
                order.append(n)
                done.add(n)
    return order, preds


class Runner:
    def __init__(self, db, wb, scope, ns, registry, use_cache=True):
        self.db, self.wb, self.scope, self.ns, self.use_cache = db, wb, scope, ns, use_cache
        self.registry = {t["tool_id"]: t for t in registry}

    def plan_calls(self, g, inputs):
        allowed = [t for t in self.registry if t in g["tools"]]  # registry order
        if llm.enabled():
            tools_txt = "\n".join(f"- {t}: {self.registry[t]['description']}" for t in allowed)
            inputs_txt = "\n".join(f"- {i.get('summary', i['hash'][:8])}" for i in inputs)
            out = llm.ask_json(AGENT_SYSTEM, f"Role: {g['role']}\nBrief: {g['brief']}\nAllowed tools:\n{tools_txt}\n"
                                             f"Inputs:\n{inputs_txt}\nAt most {g['params']['max_calls']} calls.",
                               model=g.get("model"), temperature=g["params"].get("temperature", 0.3))
            if out and isinstance(out.get("calls"), list):
                calls = [c for c in out["calls"] if c.get("tool_id") in allowed][: g["params"]["max_calls"]]
                if calls:
                    return calls, out.get("note", "")
        return [{"tool_id": t, "args": {}} for t in allowed], "default plan: allowed tools in registry order"

    def run(self, s_hash, nodes, edges, task_input, gen, stop=None):
        """nodes: {node_id: genome}. Returns (sink artifacts, cost_usd), or (None, cost) if stopped mid-structure."""
        order, preds = topo(nodes, edges)
        outputs, cost = {}, 0.0
        for nid in order:
            if stop is not None and stop.is_set():
                return None, cost
            g = nodes[nid]
            inputs = [outputs[p] for p in preds[nid]] or [task_input]
            ik = input_key(self.ns, g["agent_hash"], [i["hash"] for i in inputs])
            log(self.db, self.scope, "node_started", gen=gen, structure_hash=s_hash, node_id=nid)
            cached = self.use_cache and self.db.modules.find_one({"scope": self.scope, "input_key": ik})
            if cached:
                outputs[nid] = cached["output"]
                log(self.db, self.scope, "cache_hit", gen=gen, structure_hash=s_hash, node_id=nid,
                    trace_key=cached["trace_key"], saved_usd=cached["cost_usd"])
                continue
            calls, note = self.plan_calls(g, inputs)
            cur, trace, node_cost = inputs, [], 0.0
            for c in calls:
                t = self.registry[c["tool_id"]]
                log(self.db, self.scope, "tool_called", gen=gen, structure_hash=s_hash, node_id=nid, tool_id=t["tool_id"])
                res = self.wb.call(t["tool_id"], t["version"], c.get("args", {}),
                                   [{k: i[k] for k in ("ref", "hash", "chain") if k in i} for i in cur])
                node_cost += res.get("cost_usd", 0.0)
                trace.append({"tool_sig": tool_sig(t["tool_id"], t["version"], c.get("args", {}), [i["hash"] for i in cur]),
                              "tool_id": t["tool_id"], "args": c.get("args", {}), "summary": res.get("summary", ""),
                              "output_ref": res["output_ref"], "error": res.get("error")})
                cur = [{"ref": res["output_ref"], "hash": res["output_hash"], "chain": res.get("chain", []),
                        "summary": res.get("summary", "")}]
            out = cur[0]
            tk = trace_key(ik, [x["tool_sig"] for x in trace])
            self.db.modules.update_one({"scope": self.scope, "trace_key": tk}, {"$setOnInsert": {
                "scope": self.scope, "trace_key": tk, "input_key": ik, "agent_hash": g["agent_hash"],
                "role": g["role"], "structure_hash": s_hash, "node_id": nid, "note": note, "calls": trace,
                "output": out, "cost_usd": node_cost, "t": time.time()}}, upsert=True)
            outputs[nid] = out
            cost += node_cost
            log(self.db, self.scope, "node_done", gen=gen, structure_hash=s_hash, node_id=nid, trace_key=tk,
                calls=len(trace), cost_usd=node_cost)
        sinks = [n for n in nodes if not any(e["from"] == n for e in edges)]
        return [outputs[s] for s in sinks], cost
