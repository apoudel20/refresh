# refresh-server

The integrated backend. It wires the standalone folders into one system:

| layer | folder | role here |
|---|---|---|
| search + memory | `lineage/` | proposes agent teams (structures), gates repeats, caches nodes, remembers fitness in Atlas |
| execution | `blender-agent-harness/` | one `BlenderAgent` run per lineage node (a `claude -p` session), over the Blender MCP socket |
| scoring | `render-eval-skill/` | fast checks inside the agent loop; full eval + critic + vectors per structure |
| images | `imagegen-skill/` | agents' image tools: generate, edit, textures, UV retexture |
| dashboard | `uipack/` | talks to this API (`NEXT_PUBLIC_REFRESH_API_URL=http://localhost:8000`) |

`refresh/workbench.py` implements lineage's workbench contract with Blender; `refresh/runs.py` runs a
search per uploaded reference and translates its events for uipack; `refresh/api.py` is the FastAPI app.

```bash
cd refresh-server
uv sync                      # installs lineage, blender-agent-harness, render-eval, imagegen (editable)
uv run refresh-server        # http://localhost:8000  (lineage's own view: /lineage-ui)
uv run refresh-search ../test-assets/dog-1.png   # headless run, prints events and the best result
```

Blender must be open with the MCP extension's server started (port 9876).

## Agents run on your Claude subscription

Every agent (each lineage node and the refiner) and the main model (lineage generator) run through the Claude
Code CLI, not an API key:

- node agents: `claude -p` with `--strict-mcp-config` and one MCP server, `python -m blender_agent.mcp_tools`,
  which serves the agent's allowed tools (Blender, imagegen, `evaluate_render`) plus the built-in `Read` for
  viewing images. `blender_agent/claude_code_loop.py` turns the stream-json output into agent events and stops
  the session on the target score or a plateau. Per node, the workspace keeps `mcp.json`, `claude_cmd.json`,
  `claude_stream.jsonl` and `mcp_tools.log`.
- generator: `lineage/llm.py` with `LINEAGE_LLM=claude_code` (one `claude -p --output-format json` call).

Log in once where refresh-server runs: `claude auth login` (or `claude`, then `/login`). For a login that
doesn't depend on the keychain, run `claude setup-token` and set `CLAUDE_CODE_OAUTH_TOKEN` in `.env`.
`ANTHROPIC_API_KEY` is removed from the CLI's environment so it can't bill an API key instead.
`GET /api/health` shows `agents_ready` / `agents_auth`, and a search fails at once with a clear message
if the CLI isn't logged in. Other backends: `REFRESH_AGENT_BACKEND=anthropic|openrouter|openai`,
`REFRESH_GENERATOR_BACKEND=openrouter`.
