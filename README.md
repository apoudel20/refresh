# refresh

Long term memory using ablation on agent DAGs.

Built at the MongoDB Harness Engineering & Model Wrangling Hackathon (NYC, Sep 26 2026).
A harness proposes small teams of LLM agents to rebuild a 3D model from a reference
image, scores each team's result, and remembers what it tried. This repo is a set of
standalone folders; each one has its own README and runs on its own.

| folder | what it is |
|---|---|
| [`refresh-server/`](refresh-server/README.md) | **The integrated backend.** Runs a lineage search per reference image, executes each agent node as a `BlenderAgent` in Blender, scores teams with render-eval, serves the uipack API. |
| [`uipack/`](uipack/README.md) | Next.js dashboard: capture references, watch agent teams search (generations, team graph, scores, critic fixes, agent activity), view and pinch-sculpt the GLB. |
| [`lineage/`](lineage/README.md) | Search + memory layer (MongoDB Atlas): proposes agent structures, gates repeats with hashes and `$vectorSearch`, caches nodes, remembers fitness and outcome vectors. |
| [`blender-agent-harness/`](blender-agent-harness/README.md) | Execution layer: `BlenderAgent` tool loop (Claude Code CLI by default), Blender MCP connector (Blender Lab extension), agent runner. |
| [`render-eval-skill/`](render-eval-skill/README.md) | Scoring: 8 image evals, a VLM critic, run history and vector encodings; also an installable agent skill. |
| [`imagegen-skill/`](imagegen-skill/README.md) | Image generation, editing and UV-atlas retexturing; agents' image tools; also an installable skill. |
| `plateou-example/` | Comparison images from a run that plateaued. |
| `test-assets/` | Reference photo (`dog-1.png`), Blender renders (`p10.png`, `dog-test.png`) and control images. |
| `reports/` | A sample `render-eval` run history. |
| `.claude/skills/` | The render-eval and blender-atlas-retexture skills, installed for this repo. |

## How it fits together

1. **uipack** uploads reference images to `POST /api/reconstructions` (refresh-server).
2. **refresh-server** sets up a locked stage camera in Blender and starts a **lineage** search.
3. The **main model** (lineage generator, Claude Code CLI) proposes small teams of agents; each agent has a role, a brief and tools.
4. Each agent node runs as a **BlenderAgent**: a headless `claude -p` session on your Claude subscription whose tools are served to it over MCP (`blender_agent/mcp_tools.py`) and act on the live Blender through the MCP connector; it can use **imagegen** tools and checks itself with fast **render-eval** scores. Its scene is saved as a `.blend` snapshot and handed to the next agent.
5. Each finished team is scored with full **render-eval** (8 evals + critic) on the stage render; the composite is the fitness, and the score vector, token matrix and critique are stored in **MongoDB Atlas** (outcome vector index) for the next generation's proposals.
6. uipack streams progress and every better model (GLB), shows the search in the Agent teams panel, and sends pinch-sculpt selections back to `/refine`, which runs a refiner agent.

## Run it

```bash
cp .env.example .env                       # OPENROUTER_API_KEY (critic, embeddings), MONGODB_URI
claude auth login                          # or `claude`, then /login: agents run on your Claude subscription
# Blender 5.1+: enable the MCP extension and start its server (port 9876)
cd refresh-server && uv sync && uv run refresh-server          # API on :8000
cd uipack && npm install && cp .env.example .env.local && npm run dev   # UI on :3000
```

Install a skill into another project:

```bash
./render-eval-skill/install.sh ~/path/to/project
./imagegen-skill/install.sh ~/path/to/project
```

The skills read `OPENROUTER_API_KEY` from the environment or a `.env` file (see `.env.example`).
