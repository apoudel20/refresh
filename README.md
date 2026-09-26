# refresh

Long term memory using ablation on agent DAGs.

Built at the MongoDB Harness Engineering & Model Wrangling Hackathon (NYC, Sep 26 2026).
A harness proposes small teams of LLM agents to rebuild a 3D model from a reference
image, scores each team's result, and remembers what it tried. This repo is a set of
standalone folders; each one has its own README and runs on its own.

| folder | what it is |
|---|---|
| [`lineage/`](lineage/README.md) | MongoDB Atlas memory layer for the search over agent structures: dedupes structures, caches nodes, resumes after a crash. |
| [`render-eval-skill/`](render-eval-skill/README.md) | Scores a render against the reference image: 8 evals, a VLM critic, run history and vector encodings for comparing agent structures. Installs the `render-eval` command and agent skill. |
| [`imagegen-skill/`](imagegen-skill/README.md) | Image generation, editing and texture-atlas toolkit, plus the `blender-atlas-retexture` skill. |
| [`uipack/`](uipack/README.md) | Next.js frontend: projects, reference capture, base-model requests, GLB viewer and pinch-sculpt edits. |
| `aux/` | Scripts and renders from building the Blender dog. |
| `plateou-example/` | Comparison images from a run that plateaued. |
| `test-assets/` | Reference photo (`dog-1.png`), Blender renders (`p10.png`, `dog-test.png`) and control images. |
| `reports/` | A sample `render-eval` run history and latest-run folder. |
| `.claude/skills/` | The render-eval and blender-atlas-retexture skills, installed for this repo. |

Install a skill into another project:

```bash
./render-eval-skill/install.sh ~/path/to/project
./imagegen-skill/install.sh ~/path/to/project
```

The skills read `OPENROUTER_API_KEY` from the environment or a `.env` file (see `.env.example`).
