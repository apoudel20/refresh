# Lineage

**A MongoDB Atlas memory layer that lets a long, automated search over agent structures never repeat work.**
Built at the MongoDB Harness Engineering & Model Wrangling Hackathon, NYC, Sep 26 2026 (Statement 2: Long Horizon Engineering).

The harness proposes small DAGs of LLM agents to attack a task (first task: 3D reconstruction from an image). Each agent is defined by its traits: its model, brief and the tools it chose. The workbench executes the tool calls, and the eval engine scores each structure. Everything is written to Atlas under content hashes, so:

- **A structure that's already been tried is blocked** by its canonical hash (a unique index on the worklog).
- **A node whose agent and inputs have been seen before is reused, not rerun.** This is the node cache, Bazel/Nix style: `input_key = H(ns, agent_hash, parent output hashes)`.
- **The next generation is generated from memory:** elites, failures and untried combinations.
- **The search resumes after a crash:** rerun the same `--scope`.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (bin/pip on macOS/Linux)
cp .env.example .env    # set MONGODB_URI (hackathon Atlas sandbox) and optionally OPENROUTER_API_KEY
.venv/Scripts/python -m uvicorn lineage.server:app --port 8130          # UI at http://localhost:8130
# or headless:
.venv/Scripts/python -m lineage.search --scope demo-on --memory on --generations 6 --k 4 \
    --workbench mock            # or the workbench URL; add --eval-url for a separate eval engine
.venv/Scripts/python -m pytest -q
```

## How it works

| Piece | File |
|---|---|
| Canonical hashing (agents, structures, tool calls, node keys) | `lineage/hashing.py` |
| Generator: LLM-guided proposals from memory + mutations of elites | `lineage/generator.py` |
| Runner: DAG execution, node cache, context modules | `lineage/runner.py` |
| Search loop: propose → gate → run → evaluate → remember | `lineage/search.py` |
| Workbench / eval adapters (spec contracts) and a mock | `lineage/workbench.py` |
| Live view: change stream → server-sent events → UI | `lineage/server.py`, `ui/index.html` |

**MongoDB:**
- unique indexes as the "already tried" memory
- Atlas Vector Search: every agent and evaluated structure gets a 64-dim trait vector (feature-hashed role and tools). Before a new structure runs, `$vectorSearch` over the `struct_vec` index, filtered to the scope, skips it if it's a near-duplicate (cosine ≥ 0.97) of one already evaluated.
- change streams driving the live view
- `$graphLookup` for structure lineage
- Welford `{mean, n, var}` fitness per structure, for noisy evals

## Results: memory ON vs OFF

The same search run twice on Atlas with the same budget: 6 generations × 5 proposals, seed 1, the same generator and the same evaluator. The workbench and eval are the **mock** 3D tools: the eval rewards a hidden ideal pipeline, so the search has a real gradient to climb. All numbers come from the `events` collection.

| | memory ON (`ga-on`) | memory OFF (`ga-off`) |
|---|---|---|
| structures evaluated | 21 | 27 |
| repeats blocked by hash | 4 | 0 |
| near-duplicates skipped (`$vectorSearch`) | 2 | 0 |
| repeats run again (wasted) | 0 | 4 |
| node cache hits | 19 | 0 |
| **tool calls** | **62** | **143** |
| **best fitness** | **0.91** | **0.92** |
| best per generation | 0.39 → 0.52 → 0.57 → 0.76 → 0.91 → 0.70 | 0.39 → 0.52 → 0.53 → 0.70 → 0.92 → 0.81 |

**Memory reaches the same best fitness with 57% fewer tool calls**, and never re-runs a structure it has already tried. This is a single seed on mock tools, so treat it as a demo of the mechanism, not a benchmark.

## Built today

Everything in this repo was written on Sep 26, 2026 during the hackathon.
