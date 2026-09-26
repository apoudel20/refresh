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
- change streams driving the live view
- `$graphLookup` for structure lineage
- Welford `{mean, n, var}` fitness per structure, for noisy evals

## Built today

Everything in this repo was written on Sep 26, 2026 during the hackathon.
