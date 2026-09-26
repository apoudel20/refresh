# render-eval

Grades how closely a render of a 3D model matches a reference image, as an installable
agent skill. Eight image-to-image evals each score the pair from 0 to 1. A vision-LLM
critic lists what to fix. Every run is kept in a history file with two vector encodings,
so runs from different agent setups can be compared.

Everything lives in this folder:

| path | what it is |
|---|---|
| `render_eval/` | the Python package and the `render-eval` command |
| `skills/render-eval/SKILL.md` | the agent skill, plus `references/scores.md` |
| `install.sh` | puts the command on PATH and copies the skill into a project |
| `tests/` | offline tests (`RENDER_EVAL_SLOW_TESTS=1` also runs the model-backed ones) |
| `pyproject.toml`, `uv.lock` | dependencies |

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
./install.sh ~/path/to/project      # command on PATH + skill in <project>/.claude/skills/render-eval
./install.sh --user                 # skill for every project, in ~/.claude/skills/render-eval
./install.sh --uninstall ~/path/to/project
```

The first install downloads PyTorch, about 1 GB. The first eval run downloads about
3 GB of model weights into `~/.cache`: Marigold normals, Depth Anything V2, a
background-removal model and AlexNet for LPIPS.

The embedding step and the critic call OpenRouter. Put `OPENROUTER_API_KEY` in the
environment or in a `.env` file in the project you run from. See `.env.example`. Without
a key those two steps are skipped and the rest still run.

## Use

From the project root:

```bash
render-eval critique refs/reference.png renders/latest.png --label "rebuilt the ears" \
  --meta '{"structure": "geometry+fur"}'
```

Each run does three things:

1. **Appends the run** to `reports/report.json`: scores, raw metrics, critic output,
   label, meta, image hashes and vectors.
2. **Replaces `reports/latest-run/`** with this run's images: `overview.png` with every
   step, one image per step, `report.html`, `critique.md`, `run.json` and `vector.json`.
3. **Prints the critique**: the composite score, the change since the previous run on the
   same reference, the critic's notes and top fixes, a verdict per step, and the most
   similar earlier run.

Other commands:

```bash
render-eval vectors [--mode scores] [--delta] [--export v.json]   # compare recorded runs
render-eval compare ref.png a.png b.png          # which render is closer, asked both ways round
render-eval report ref.png a.png b.png -o out/   # side-by-side HTML report, no history
render-eval run ref.png render.png --json        # one-off scores, no history
render-eval list                                 # the eight evals and their weights
```

Or from Python:

```python
from render_eval import record_run, run_evals, encode

res = record_run("ref.png", "render.png", "reports", label="round 3")
print(res["critique"], res["vector"].tokens.shape)   # (8, 32)
```

## The evals

| # | eval | measures | score |
|---|---|---|---|
| 1 | pixel | PSNR, SSIM, LPIPS | 0.5 SSIM + 0.5 (1 − LPIPS) |
| 2 | depth | Depth Anything V2 depth maps, scale- and shift-invariant | Spearman correlation |
| 3 | normals | Marigold surface normals | share within 22.5° |
| 4 | silhouette | object mask overlap and outline distance | IoU |
| 5 | edges | Canny edges matched with a tolerance | edge F1 |
| 6 | embedding | Voyage multimodal embeddings via OpenRouter | cosine |
| 7 | color | shared CIELAB palette, earth mover's distance | exp(−EMD / 20) |
| 8 | judge | vision LLM rubric via OpenRouter (default Claude Opus 5.5) | mean criterion, rescaled |

Each run is encoded as a score vector (8 numbers) and a token matrix (8 x 32). A token
holds its step's metrics and a 4 x 4 grid of where the render differs.
`skills/render-eval/references/scores.md` documents every slot.

Reference numbers, `dog-1.png` against:

| candidate | composite | critic |
|---|---|---|
| itself | 1.00 | 10/10 |
| blurred copy | 0.81 | 7/10 |
| mirrored copy | 0.59 | 9/10 |
| Blender render p10 | 0.57 | 5 to 6/10 |
| Blender render dog-test | 0.47 | 5/10 |
| a different dog | 0.41 | 3/10 |
| random noise | 0.22 | 1/10 |

## Develop

```bash
uv run pytest                          # fast, offline
RENDER_EVAL_SLOW_TESTS=1 uv run pytest     # also runs LPIPS, depth and normals models
```
