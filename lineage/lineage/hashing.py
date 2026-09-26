"""Canonical hashing: the identity layer for agents, structures and node runs (spec §6)."""
import hashlib
import json
import unicodedata

VOLATILE = {"timestamp", "ts", "time", "request_id"}
SET_FIELDS = ("tools", "delegates")
MAX_NODES, MAX_DEPTH = 6, 3


def _norm(x):
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, str):
        return unicodedata.normalize("NFC", x).strip()
    return x


def canon(x) -> str:
    return json.dumps(_norm(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def H(*parts) -> str:
    return hashlib.sha256(canon(list(parts)).encode()).hexdigest()


def content_hash(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()


def normalize_genome(genome: dict) -> dict:
    g = {k: v for k, v in genome.items() if k != "agent_hash"}
    for f in SET_FIELDS:
        g[f] = sorted(set(g.get(f, [])))
    return g


def agent_hash(genome: dict) -> str:
    return H("agent", normalize_genome(genome))


def namespace(task_id, task_input_hash, eval_version, tools_version) -> str:
    return H("ns", task_id, task_input_hash, eval_version, tools_version)


def validate(nodes: dict, edges: list) -> None:
    """nodes: {node_id: agent_hash}; edges: [{from, to, type, order}]. Raises on invalid DAGs."""
    if not 1 <= len(nodes) <= MAX_NODES:
        raise ValueError(f"structure needs 1..{MAX_NODES} nodes")
    if len(set(nodes.values())) != len(nodes):
        raise ValueError("an agent may appear at most once per structure")
    preds = {n: [] for n in nodes}
    for e in edges:
        if e["from"] not in nodes or e["to"] not in nodes:
            raise ValueError(f"edge references unknown node: {e}")
        preds[e["to"]].append(e["from"])
    roots = [n for n, p in preds.items() if not p]
    if len(roots) != 1:
        raise ValueError("structure needs exactly one root")
    depth = {}

    def d(n, seen=()):
        if n in seen:
            raise ValueError("cycle")
        if n not in depth:
            depth[n] = 1 + max((d(p, seen + (n,)) for p in preds[n]), default=0)
        return depth[n]

    if max(d(n) for n in nodes) > MAX_DEPTH:
        raise ValueError(f"depth over {MAX_DEPTH}")


def structure_hash(ns: str, nodes: dict, edges: list) -> str:
    validate(nodes, edges)
    e = sorted((nodes[x["from"]], nodes[x["to"]], x.get("type", "parallel"),
                x.get("order", 0) if x.get("type") == "then" else 0) for x in edges)
    return H("structure", ns, sorted(nodes.values()), e)


VEC_DIMS = 64


def _unit(v):
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [round(x / n, 6) for x in v]


def trait_vec(role, tools) -> list:
    """An agent's traits (role, tools) feature-hashed into 64 buckets, L2-normalized: the space $vectorSearch compares in."""
    v = [0.0] * VEC_DIMS
    for f in [f"role:{role}"] + [f"tool:{t}" for t in sorted(set(tools))]:
        v[int(hashlib.sha256(f.encode()).hexdigest(), 16) % VEC_DIMS] += 1.0
    return _unit(v)


def structure_vec(agents) -> list:
    """Mean trait vector of a structure's agents ([{role, tools}]), re-normalized for cosine."""
    vs = [trait_vec(a["role"], a["tools"]) for a in agents]
    return _unit([sum(col) / len(vs) for col in zip(*vs)])


def tool_sig(tool_id, version, args, input_hashes) -> str:
    return H("tool", tool_id, version, args, sorted(input_hashes))


def input_key(ns, a_hash, parent_output_hashes) -> str:
    return H("input", ns, a_hash, sorted(parent_output_hashes))


def trace_key(in_key, sigs) -> str:
    return H("trace", in_key, list(sigs))
