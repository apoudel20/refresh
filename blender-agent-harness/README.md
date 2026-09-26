# blender-agent-harness

Execution layer for Refresh: one `BlenderAgent` run is one agent node in a lineage structure.

| package | what it is |
|---|---|
| `blender_mcp_connector/` | Talks to Blender over the MCP socket (default: the Blender Lab MCP extension, port 9876; `BLENDER_MCP_PROTOCOL=community` for the community add-on). Also owns the locked stage camera, node snapshots and GLB export. |
| `blender_agent/` | The agent loop (`agent.py`; Claude Code CLI loop in `claude_code_loop.py` with its tools served over MCP by `mcp_tools.py`; Anthropic API loop in `anthropic_loop.py`), tools (`tools.py`), render-eval scoring backend (`evaluator.py`), imagegen tools (`imagegen_tools.py`), the solidity check (`solidity.py`), the reference-photo guard (`reference_guard.py`), event emitters. |
| `agent_runner/` | `run(ModelConfig(), reference_image=...)` spins up a single agent outside the search. |

Run inside the integrated project (see `../refresh-server`), or alone:

```python
from agent_runner import ModelConfig, run
result, trace = run(ModelConfig(model="claude-opus-5", backend="anthropic"), reference_image="ref.png")
```

`dashboard.py` (Streamlit) is superseded by the uipack dashboard; install the `dashboard` extra to use it.
