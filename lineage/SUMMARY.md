# Lineage: summary for the team

**Lineage is the memory and search layer.** It runs a genetic algorithm over small pipelines of LLM agents ("structures") and stores everything in MongoDB Atlas, so a long search never pays for the same work twice. It doesn't do 3D itself. It calls a **workbench** for tools and an **eval** for scores, and those are yours.

## What it does, in one loop
1. **Propose** a generation of structures. Each is a DAG of 1–6 agents, and each agent has a role and a set of allowed tools. They come from LLM proposals plus mutations of the best-scoring structures so far.
2. **Gate** each one before running it, three ways:
   - **Exact repeat:** its content hash is already in Atlas (unique index), so it's blocked.
   - **Near-duplicate:** Atlas `$vectorSearch` over 64-dim trait vectors finds one already evaluated with cosine ≥ 0.97, so it's skipped.
   - **Node cache:** an agent that has already seen these exact inputs reuses its stored output instead of rerunning.
3. **Run** the DAG. Each agent picks tool calls, and the workbench executes them.
4. **Evaluate** the final outputs to get a fitness in [0, 1], stored as a running mean, n and variance.
5. **Remember:** the best structures seed the next generation. Rerunning the same `--scope` resumes the search after a crash.

A live UI (FastAPI plus a MongoDB change stream) shows each generation as cards, the running DAG, "reused from memory" nodes, blocked and near-duplicate ghosts, and the best-fitness curve.

## Status (Sep 26, 4:20 PM)
Everything below runs on the **hackathon Atlas cluster**, all pushed to `main` under `lineage/`:
- the hashing gates, node cache, resume, change-stream UI and `$graphLookup` lineage
- the `$vectorSearch` near-duplicate gate (indexes `agent_vec` and `struct_vec`)
- the genetic algorithm climbs on the mock task: best fitness 0.39 → 0.91 over 5 generations

**Memory ON vs OFF** (same budget, mock tools):

| | ON | OFF |
|---|---|---|
| best fitness | 0.91 | 0.92 |
| tool calls | **62** | **143** |
| wasted repeats | 0 | 4 |
| cache hits | 19 | 0 |

The ON run reaches the same score with 57% fewer tool calls. Full table in `lineage/README.md`.

**Not wired yet:** the real Blender workbench and the real render eval. The demo uses mock tools and a mock eval, and we say so.

## Run it
```
cd lineage
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# .env: MONGODB_URI=<Atlas>, optional OPENROUTER_API_KEY; MOCK_DELAY=0.4 makes the mock watchable
.venv/Scripts/python -m uvicorn lineage.server:app --port 8130     # UI at http://localhost:8130
.venv/Scripts/python -m lineage.search --scope my-run --memory on --generations 6 --k 5 --workbench mock
```

## How your parts plug in (the contract Lineage calls)
**Workbench:** Lineage calls it for every tool call.
- `GET /tools` returns `[{tool_id, version, description}]`
- `POST /call {tool_id, version, args, inputs: [{ref, hash}]}` returns `{output_ref, output_hash, summary, cost_usd, error}`
  - `output_ref` can be a file path in a shared workspace.
  - `output_hash` should be a content hash of the output file, because that's what makes the node cache work.

**Eval:** Lineage calls it once per structure with that structure's final outputs, and needs a single number.
- It needs `fitness` in [0, 1]. Your `eval_api.py` already returns `overall_score` in [0, 1], which is exactly what we need.

## What's between us and the real product (the hops)
1. ✅ **Eval adapter: done and tested.** When the final outputs are PNG paths, `HttpEval` sends them as multipart to your `POST /eval`, adding `EVAL_REFERENCE_IMAGE` if set, with `backend=openrouter` unless `EVAL_BACKEND` says otherwise. It maps `overall_score` to `fitness` and keeps the sub-scores as metrics.
   Tested against your `eval_api.py` (run with `PYTHONPATH=blender_agent/imagegen-skill;blender_agent/render-eval-skill`) on `plateou-example/cmp1.png`: fitness 0.70.
   To use it: `--eval-url http://<host>:8140`, or the eval-URL field in the UI.
2. **Workbench over HTTP (your side, needs Blender).** Wrap `TOOL_DEFINITIONS` and `BlenderAgent._call_tool` (the MCP connector) in a tiny FastAPI with the `/tools` and `/call` above. It must run on a machine with Blender and the MCP add-on.
3. **Agent = your `BlenderAgent`.** Blender tools need real arguments (paths, object names, code), which your agent loop already produces. The cleanest mapping is: one Lineage node = one `BlenderAgent` run with `AgentTraits` restricted to that node's role and tools. `AgentTraits` is already the "genome" Lineage hashes.
4. **Artifacts through a shared workspace.** Each node's output (PLY, blend or renders) is a path plus its content hash, and it becomes the input to the next node.
5. **Budget.** A real structure takes minutes and costs real money, so run small generations (k = 3–4) and lean on the cache.

Hop 1 is done. Hops 2 and 3 need someone with Blender running. Once they exist, hops 4 and 5 are small Lineage changes.
