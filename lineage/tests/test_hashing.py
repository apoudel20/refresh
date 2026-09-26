import pytest

from lineage.hashing import agent_hash, structure_hash, structure_vec, trace_key, trait_vec

NS = "ns"
A, B, C, D = "a", "b", "c", "d"


def test_redrawn_dag_hashes_the_same():
    s1 = structure_hash(NS, {"n0": A, "n1": B, "n2": C},
                        [{"from": "n0", "to": "n1", "type": "parallel"}, {"from": "n0", "to": "n2", "type": "parallel"}])
    s2 = structure_hash(NS, {"x": A, "y": C, "z": B},
                        [{"from": "x", "to": "y", "type": "parallel"}, {"from": "x", "to": "z", "type": "parallel"}])
    assert s1 == s2


def test_shared_child_differs_from_chain():
    diamond = structure_hash(NS, {"n0": A, "n1": B, "n2": C, "n3": D},
                             [{"from": "n0", "to": "n1"}, {"from": "n0", "to": "n2"},
                              {"from": "n1", "to": "n3"}, {"from": "n2", "to": "n3"}])
    tree = structure_hash(NS, {"n0": A, "n1": B, "n2": C, "n3": D},
                          [{"from": "n0", "to": "n1"}, {"from": "n0", "to": "n2"}, {"from": "n1", "to": "n3"}])
    assert diamond != tree


def test_then_order_matters_parallel_order_does_not():
    e1 = [{"from": "n0", "to": "n1", "type": "then", "order": 1}, {"from": "n0", "to": "n2", "type": "then", "order": 2}]
    e2 = [{"from": "n0", "to": "n1", "type": "then", "order": 2}, {"from": "n0", "to": "n2", "type": "then", "order": 1}]
    nodes = {"n0": A, "n1": B, "n2": C}
    assert structure_hash(NS, nodes, e1) != structure_hash(NS, nodes, e2)


def test_agent_hash_ignores_set_order_and_whitespace():
    g1 = {"role": "refiner", "brief": " Refine. ", "tools": ["b", "a"], "delegates": [], "params": {"t": 0.30000001}}
    g2 = {"role": "refiner", "brief": "Refine.", "tools": ["a", "b"], "delegates": [], "params": {"t": 0.3}}
    assert agent_hash(g1) == agent_hash(g2)


def test_rejects_duplicate_agent_and_multiple_roots():
    with pytest.raises(ValueError):
        structure_hash(NS, {"n0": A, "n1": A}, [{"from": "n0", "to": "n1"}])
    with pytest.raises(ValueError):
        structure_hash(NS, {"n0": A, "n1": B}, [])


def test_tool_order_matters_in_trace():
    assert trace_key("k", ["s1", "s2"]) != trace_key("k", ["s2", "s1"])


def test_trait_vectors_are_unit_and_order_free():
    v = trait_vec("geometry", ["coarse_mesh", "depth_estimate"])
    assert len(v) == 64 and abs(sum(x * x for x in v) - 1) < 1e-4
    assert v == trait_vec("geometry", ["depth_estimate", "coarse_mesh", "coarse_mesh"])


def test_structure_vec_ignores_brief_and_node_order():
    a, b = {"role": "planner", "tools": ["segment"], "brief": "x"}, {"role": "refiner", "tools": ["mesh_refine"]}
    assert structure_vec([a, b]) == structure_vec([b, {**a, "brief": "y"}])
    assert structure_vec([a]) != structure_vec([b])
