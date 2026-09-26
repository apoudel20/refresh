# imagegen

A toolkit for generating, editing, upscaling and texture-atlasing images with AI image models.
Use it as a CLI (`imagegen`), as a Python library, or as an MCP server (`imagegen-mcp`) that Codex,
Claude Code and other agents can call.

It ships with an agent skill, [`blender-atlas-retexture`](skills/blender-atlas-retexture/SKILL.md),
that retextures a Blender object's UV texture atlas from a reference image and checks the result
against texture-atlas rules.

## Install

Requirements:
- [uv](https://docs.astral.sh/uv/)
- Blender, for the `uv` commands
- One image backend: the [Codex CLI](https://github.com/openai/codex) logged in with `codex login`,
  or an `OPENROUTER_API_KEY`

```bash
./install.sh /path/to/your/project            # CLI + skill into <project>/.claude/skills
./install.sh /path/to/your/project --mcp      # also register the MCP server in <project>/.mcp.json
./install.sh /path/to/your/project --codex    # also copy the skill to ~/.codex/skills
```

What `install.sh` does:
1. Installs the `imagegen` and `imagegen-mcp` commands on your PATH as a uv tool.
2. Copies the skill into the target project.
3. With the flags above, registers the MCP server and/or installs the skill for Codex.
4. Reports whether Blender and the backends are usable.

Re-run it after pulling changes. `./install.sh --uninstall /path/to/project` removes it again.

## Backends

| backend | billing | how it works |
|---|---|---|
| `codex` (default when installed) | your ChatGPT/Codex subscription | runs `codex exec` with its built-in image tool and reads results from `~/.codex/generated_images/<thread>/` |
| `openrouter` | `OPENROUTER_API_KEY` (pay per image) | `POST /api/v1/images`; default `google/gemini-3.1-flash-image`, any model from `imagegen models` |

Pick one with `--backend` or `IMAGEGEN_BACKEND`, and a model with `--model`.

- **Codex model.** The default is `IMAGEGEN_CODEX_MODEL`, or `gpt-5.5` if unset. Newer models such as
  `gpt-6-astra` need a newer CLI than 0.142.5 (`codex update`).
- **Speed.** Each Codex call takes about 40s, and calls run in parallel (`--concurrency`, default 4).
- **Isolation.** The Codex sub-agent runs `--ephemeral`, with a read-only sandbox, in a temp dir, and
  without your `config.toml`. Your MCP servers and rules aren't loaded; login still is.
- **AI review.** `atlas review` uses the same backend as a vision judge: Codex with `--output-schema`,
  or OpenRouter chat with `google/gemini-3.8-flash`.

```bash
imagegen backends          # what's usable right now (spends nothing)
```

## Single images

```bash
# generate (exact size = generate, then crop/resize)
imagegen generate "hand-painted mossy cobblestone, seamless tileable, top-down" -o out/cobble.png --size 512
imagegen generate "red potion bottle on flat #ff00ff background" -o out/potion.png --size 256 --transparent '#ff00ff'
imagegen generate "same knight, facing left" --ref knight.png -o out/knight-left.png

# edit the whole image, or only a region (x,y,w,h px, or fractions) or mask (white = editable)
imagegen edit out/cobble.png "add puddles" -o out/cobble-wet.png
imagegen edit out/cobble.png "replace with a rusty iron grate" --region 0.3,0.3,0.4,0.4 -o out/grate.png
imagegen edit photo.png "remove the sign" --mask sign-mask.png -o fixed.png

# upscale: lanczos/bicubic/nearest are local and free; ai re-details overlapping tiles
imagegen upscale out/cobble.png -s 4 --method ai --hint "stone floor texture" -o out/cobble-4x.png
imagegen upscale sprite.png -s 8 --method nearest -o sprite-8x.png

# make a texture tile seamlessly (blend = local; ai = model repaints the seams)
imagegen seamless tex.png -o tex-tile.png --method ai --preview tex-2x2.png
imagegen seam-score out/*.png            # ~1.0 = invisible seam, > 1.6 = visible

imagegen key sprite.png -o sprite-alpha.png --color '#ff00ff'
```

**How region edits stay exact.** The model receives a context crop around the region (sized to an
aspect ratio it supports) plus a copy with the region highlighted. Its output is colour-matched on
the ring just outside the region, which cancels the model's global colour drift. It is then
composited back through a mask feathered only inward. Pixels outside the region are bit-identical to
the input.

## Texture atlases

An atlas is a project directory. `cells/*.png` is the source of truth, and `atlas.png` and
`atlas.json` are rebuilt from it.

```bash
imagegen atlas init atlas.json                  # example spec (or --example sprites)
# edit atlas.json: style, cell_size, tileable, background, cells[{name, prompt}]
imagegen atlas build atlas.json -o assets/dungeon   # resumable; re-run to fill failed cells

imagegen atlas review assets/dungeon            # vision model scores each cell 0-10 + style outliers
imagegen atlas review assets/dungeon --fix      # edit (score 4-6) / regenerate (<= 3) failures, re-review

imagegen atlas cell assets/dungeon lava --edit "make the cracks thinner"
imagegen atlas cell assets/dungeon lava --edit "add a skull" --region 0.3,0.3,0.4,0.4
imagegen atlas cell assets/dungeon lava --prompt "cooled obsidian with faint red glow"   # regenerate
imagegen atlas cell assets/dungeon lava --image my-lava.png                             # hand-made
imagegen atlas revert assets/dungeon lava       # every change is kept in history/

imagegen atlas upscale assets/dungeon -s 2 --method ai   # per cell, so the grid stays exact
imagegen atlas repack assets/dungeon --padding 8 --pot

# bring an existing atlas in, then edit it cell by cell
imagegen atlas import old_atlas.png --grid 8x8 --source-padding 2 -o assets/old
imagegen atlas slice sheet.png --grid 4x4 -o cells/
imagegen atlas pack cells/*.png -o packed.png --padding 4
```

Spec fields (`AtlasSpec` in `atlas.py`):

| field | default | meaning |
|---|---|---|
| `cell_size` | 256 | `N` or `[w, h]` px per cell |
| `columns` | √n | grid width |
| `padding` | 4 | extruded px around each cell (wrap for tileable, clamp otherwise) to stop mip/filter bleed |
| `style` | "" | shared art direction in every prompt |
| `tileable` | false | default for cells; tileable cells get seam rules, auto seam-fix, wrap padding |
| `background` | opaque | `transparent` = generate on `key_color`, then chroma-key |
| `mode` | cells | `cells`: one call per cell, most accurate. `sheet`: whole grid in one call from a numbered layout guide, most consistent |
| `style_reference` | — | image every cell must match stylistically |
| `chain_style` | true | with no `style_reference`, the first finished cell becomes the style anchor for the rest |
| `resample` | lanczos | `nearest` for pixel art |
| `power_of_two` | false | pad `atlas.png` to POT |

`atlas.json` uses TexturePacker's "JSON hash" layout. Each frame has `frame`, `sourceSize`, `uv`
(top-left origin), `uv_gl` (bottom-left origin), `col`, `row`, `tileable` and `prompt`.

## Blender UV atlases (retexture a model from a reference image)

`imagegen uv ...` repaints the base-colour texture of a UV-unwrapped Blender object so it looks
like a reference image. It keeps the UV layout intact and verifies the result. The step-by-step
agent workflow is the project skill
[`skills/blender-atlas-retexture`](skills/blender-atlas-retexture/SKILL.md).

```bash
imagegen uv demo -o work/demo                       # try it: UV-unwrapped house + flat atlas
imagegen uv export scene.blend --object House -o work/uv   # islands, normals, UV rotation, seams
imagegen uv render scene.blend --object House -o work/renders/before
imagegen uv retexture --uv work/uv --style ref.png -o work/atlas_v1.png \
    -M up="terracotta roof tiles" -M side="fieldstone wall" -M rest="dark stone"
imagegen uv render scene.blend --object House --image work/atlas_v1.png -o work/renders/v1
imagegen uv review-renders --after work/renders/v1 --before work/renders/before --style ref.png
```

How the model is kept in line:

- **Masked passes.** Each (material × UV rotation) is one masked edit, painted on an atlas turned
  so that "up" on the model is up.
- **Clamping and bleed.** Only in-island texels are kept, and island colours are bled 16 px into
  the padding.
- **Checks** (`uv check`): size, coverage, bleed, drift against Blender's island outlines, seam
  tone, untouched empty space, and UV sanity. A vision review is optional. Failing islands are
  repaired with masked edits.
- **Final verification** is a Blender render reviewed against the reference.

## Agents (MCP)

`imagegen-mcp` starts a stdio MCP server with 22 tools: `generate_image`, `edit_image`,
`upscale_image`, `make_seamless`, `chroma_key`, `imagegen_status`, and `atlas_*` (init_spec, build,
edit_cell, revert_cell, review, repack, upscale, import, pack, slice), and `uv_*` (export,
retexture, check, repair, render, review_renders). Image tools return a small
preview, so the agent can see what it made.

- **Claude Code:** `./install.sh <project> --mcp` adds it to `<project>/.mcp.json`.
- **Codex:** add this to `~/.codex/config.toml`:

  ```toml
  [mcp_servers.imagegen]
  command = "imagegen-mcp"
  tool_timeout_sec = 1800   # atlas builds/reviews make several ~40s model calls
  ```

## Python

```python
from imagegen import get_backend, generate, edit, upscale, seamless, AtlasSpec
from imagegen import atlas

b = get_backend("codex")
[img], _ = generate("mossy cobblestone, seamless", backend=b, size=(512, 512))
img = edit(img, "add a drain grate", backend=b, region=(192, 192, 320, 320)).image
atlas.build(AtlasSpec.load("atlas.json"), "assets/dungeon", b)
atlas.review("assets/dungeon", b, fix=True)
```

## Development

```bash
uv sync
uv run pytest        # offline: fake backend, fake `codex` executable, mocked OpenRouter HTTP
```

The Blender round-trip test runs when Blender is installed and is skipped otherwise.
