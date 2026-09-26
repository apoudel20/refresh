---
name: render-eval
description: Score a render of a 3D or Blender model against a reference image and get a critic's list of fixes, a run history, and vector encodings for comparing runs. Use after rendering the model to a PNG while iterating toward a reference photo; when asked to evaluate, grade, score, compare or critique a render against a reference; when deciding whether a round of changes (or an agent team's work) improved or regressed the model; or when comparing runs from different agent structures. Records every run in reports/report.json and replaces reports/latest-run/ with the comparison images.
---

# Render eval

Compares a render of the model with the reference image in eight steps and prints a
critique. The steps are pixel similarity, depth, surface normals, silhouette, edges,
image-embedding similarity, color, and a vision-LLM critic. The critique holds the
composite score (0 to 1, higher is closer), the change since the previous run, the
critic's notes per criterion, its top fixes, and one verdict line per step.

The `render-eval` command comes from this skill's installer. If it is not found, tell
the user to run `install.sh` from the render-eval-skill folder.

## Run it

1. Render the current model to a PNG. Match the reference's camera angle and framing,
   and use a transparent background (Film > Transparent) so the object mask comes from
   the alpha channel. Keep camera, lights and render settings fixed between runs, or the
   scores measure the setup instead of the model.
2. From the project root, run:

   ```bash
   render-eval critique path/to/reference.png path/to/render.png --label "what changed"
   ```

   Add `--meta '{"structure": "geometry+fur+materials"}'` to tag the run with the agent
   structure or configuration that produced it.
3. Read the printed critique. Work through "Top fixes" in order. After the next render,
   run it again and check that the composite and the step you targeted went up.

A run takes 15 to 30 seconds. The first run on a machine downloads about 3 GB of models.

## Deciding whether a round helped

- Re-scoring an unchanged render moves the composite by about 0.01, mostly from the
  critic. Treat changes within about ±0.02 as no change.
- Reject a round that raises the composite while the critic's shape score falls. That
  pattern usually means a metric was gamed, for example by moving the camera.
- For a close call between two renders, ask the critic directly:
  `render-eval compare reference.png render_a.png render_b.png`. It asks twice with the
  order swapped and reports P(A closer).

## What it writes

- stdout: the critique text, also saved as `reports/latest-run/critique.md`.
- `reports/report.json`: every run recorded in this project, appended, with all
  scores, metrics, critic output, label, meta and vectors.
- `reports/latest-run/`: the latest run only. It is deleted and rewritten on every run.
  - `overview.png`: every step's output in one image. Look at this first.
  - `00-aligned.png`, `01-pixel.png`, `02-depth.png`, `03-normals.png`,
    `04-silhouette.png`, `05-edges.png`, `06-embedding.png`, `07-color.png`,
    `08-tokens.png`.
  - `reference.png`, `candidate.png`, `report.html`, `run.json`, `vector.json`.

## Comparing runs as vectors

Each run is stored as a score vector (8 numbers, one per step) and a token matrix
(8 tokens x 32 numbers, one token per step: its key metrics plus a 4 x 4 grid of where
the render differs). Compare runs, for example to see which agent structures behave alike:

```bash
render-eval vectors                        # distance matrix over token matrices
render-eval vectors --mode scores          # same, using the 8 step scores
render-eval vectors --delta                # compare what each run changed
render-eval vectors --export vectors.json  # vectors with slot names
```

Only compare runs against the same reference. See `references/scores.md` for what every
step and every token slot means.

## Options

- `--evals silhouette,depth,normals,judge`: run a subset, which is faster and cheaper.
- `--normals-backend depth`: skip the Marigold normal model, which is faster.
- `--judge-model <openrouter model id>`: change the critic model.
- `--reports-dir <dir>`: write somewhere other than `./reports`.
- `--json`: print the run record as JSON instead of the critique text.
- `render-eval report reference.png a.png b.png c.png -o out/`: one HTML report
  comparing several candidates side by side, without touching the run history.

## Requirements

- The embedding and critic steps call OpenRouter and need `OPENROUTER_API_KEY`, from
  the environment or a `.env` file in the project. Without it they are skipped and the
  composite uses the other six steps. The critic costs about $0.02 per run.
- Scores are relative. Compare runs with each other rather than reading absolutes.
  Silhouette overlap is about 0.5 even for unrelated shapes.
