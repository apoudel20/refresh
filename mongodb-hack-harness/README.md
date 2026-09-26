# mongodb-hack-harness

- built with MongoDB Atlas, Vector Search, and Agentic Memory tooling

Tools and assets for getting an agent-built Blender model to match a reference image.
Nothing at this level runs on its own; each tool is a self-contained folder with its own
installer.

| folder | what it is |
|---|---|
| [`render-eval-skill/`](render-eval-skill/README.md) | Scores a render against a reference image: 8 evals, a VLM critic, run history, vector encodings. Installs the `render-eval` command and agent skill. |
| [`imagegen-skill/`](imagegen-skill/README.md) | Image generation, editing and texture-atlas toolkit, plus the `blender-atlas-retexture` skill. |
| `aux/` | Scripts and renders from building the Blender dog. |
| `plateou-example/` | Comparison images from a run that plateaued. |
| `test-assets/` | Reference photo (`dog-1.png`), Blender renders (`p10.png`, `dog-test.png`) and control images. |
| `reports/` | History and latest run written by `render-eval` in this folder. |
| `.claude/skills/` | Both skills, installed for this folder. |

```bash
./render-eval-skill/install.sh ~/path/to/project
./imagegen-skill/install.sh ~/path/to/project
```

Both read `OPENROUTER_API_KEY` from the environment or a `.env` file (see `.env.example`).
