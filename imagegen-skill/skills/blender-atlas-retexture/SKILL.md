---
name: blender-atlas-retexture
description: Retexture a Blender object's UV texture atlas so it looks like a reference image, using an AI image model (the user's Codex/ChatGPT subscription by default, or OpenRouter), then verify the result against Blender texture-atlas rules and on renders of the actual mesh. Use this whenever someone wants to texture, re-texture, restyle, "make it look like this photo", improve, or generate a texture/UV map/texture atlas/albedo/base-color map for a .blend object or 3D model from a reference image or description — including when they only say the current texturing is "bad", "flat", "placeholder" or "not good enough", or ask to "paint", "skin" or "material" a mesh from a picture. Also use it to check or fix an existing texture atlas for UV-island drift, missing padding/bleed, seams, or wrong materials on parts of a model.
---

# Retexture a Blender object's UV atlas from a reference image

The atlas is a contract with the mesh: every texel is glued to one spot on the surface through
the UV islands. Image models don't know that contract. Left alone they shift and rescale the
layout, paint across island borders, put the wrong material on a part, paint patterns sideways on
islands that are rotated in UV space, bake lighting into the texture, and ignore seams. So this
workflow never trusts the model's pixels directly:

- **Constrain.** The model repaints the atlas, but only texels inside UV islands are kept.
  Everything else stays original, and island colours are bled into the padding. Each surface type
  is painted in its own masked pass, on an atlas turned so that "up" on the model is up in the image.
- **Verify.** Every result is checked against the atlas rules, then rendered in Blender and
  reviewed against the reference. Expect errors on the first pass. Finding and fixing them is part
  of the job, not a sign something broke.

The tools come from the `imagegen` package; its `install.sh` puts the `imagegen` and
`imagegen-mcp` commands on PATH. `imagegen uv --help` lists them, and each has `--json` for
machine-readable output. If `imagegen` isn't found, tell the user to run `install.sh` from the
imagegen repo.

## 0. Preconditions

```bash
imagegen backends     # which image backend is usable (spends nothing)
```

- **Backend.** `codex` (the default when installed) runs `codex exec` with Codex's built-in image
  tool on the user's ChatGPT/Codex plan. Each call takes about 30–90 s. `openrouter` needs
  `OPENROUTER_API_KEY` and is pay-per-image. Pass `--backend` to choose explicitly. If Codex says a
  model "requires a newer version of Codex", pass `-m gpt-5.5` or ask the user to run `codex update`.
- **Blender.** Must be on PATH, at `/Applications/Blender.app`, or set via `$BLENDER`. The scripts
  run Blender in the background and never modify the user's .blend.
- **Inputs.** You need the `.blend`, the object name, and one or more style reference images.
  - If the user describes the look in words, make a reference first:
    `imagegen generate "<photo of the look>" -o ref.png --size 768`. Show it to them before
    retexturing, because it steers everything.
  - The current atlas is found automatically: the Image Texture node on the object's material.
- **Scope.** This workflow is for the **base colour / albedo** map only. Never run it on normal,
  roughness, metallic or other non-colour maps. Their pixels are data, and an image model will
  destroy them.

## 1. Export the UV layout

```bash
imagegen uv export scene.blend --object "House" -o work/uv
```

This writes `work/uv/uv_data.npz` (island id, world normal and UV rotation per texel),
`uv_info.json` and `uv_preview.png` (numbered islands). Read the summary and **look at
`uv_preview.png`**:

- **textures**: which image feeds Base Color, and its colour space (should be sRGB). If there are
  several image nodes or UV maps, pick the base-colour one (`--uv-map` or `--size` if needed).
  If none exists, pass `--size 2048` and start from `uv placeholder`, which makes flat per-surface
  colours.
- **islands**:
  - `faces` gives the fraction of each island's texels that `up`, `side` and `down` would select.
    Use it, not the island's average `normal`. A half-wall, half-roof island averages to "slightly
    up" (z≈0.38) even though most of it is wall.
  - `up_rotation` shows how the island is turned in UV space: `{"270": 1.0}` means world-up points
    left in the image. Mixed rotations are normal and are handled automatically.
- **sanity**: report problems to the user. They are UV-layout issues that the texture can't fix:
  - overlapping islands: mirrored or stacked UVs get identical paint
  - UVs outside 0–1: the texture repeats there
  - flipped faces: patterns and text appear mirrored
  - high `texel_density_cv`: pattern scale will differ across parts
  - UDIM: do one tile at a time

## 2. Render "before"

```bash
imagegen uv render scene.blend --object "House" -o work/renders/before --opaque
```

Read a couple of the PNGs, so you and the user know what "better" has to beat. Use `--opaque`
(grey background) for anything a person will look at; transparent PNGs show as black or white
depending on the viewer.

## 3. Plan materials (the most important decision)

Decide which parts of the model get which material, and express that as `--material SELECTOR=description`.
Selectors pick texels using Blender's per-texel data:

| selector | picks | typical use |
|---|---|---|
| `up` / `down` / `side` | surface facing up (normal.z > 0.35) / down / sideways | roof vs walls vs underside; tabletop vs legs |
| `+x -x +y -y +z -z` | dominant facing axis | front vs back of a sign, one side of a crate |
| `islands:1,4` | UV islands by number (from `uv_preview.png`) | character parts, labelled panels |
| `rest` / `all` | everything not yet assigned / everything | a catch-all last entry |

- **Order and coverage.** Earlier entries win. Finish with a `rest=` entry so nothing is left
  unpainted; unassigned texels keep the old texture, and the tool warns about them.
- **Mixed islands are fine.** Selectors work per texel, so an island that holds both a wall and a
  roof gets both materials.
- **Describe materials concretely.** "Weathered terracotta barrel roof tiles in rows" beats "roof".
  The description plus the style image is the whole brief.
- **Organic objects** (characters, rocks) where material doesn't follow orientation: use
  `islands:` selectors. If it really is one material, use a single `all=...`.
- **Why passes beat one whole-atlas call.** A single call has to infer which part is which, and it
  paints every pattern in the same 2D direction. In testing it put roof tiles on gable walls and
  ran stones vertically on UV-rotated walls. Material passes remove both failure modes.

## 4. Retexture (constrain → check → repair)

```bash
imagegen uv retexture --uv work/uv --style ref.png -o work/atlas_v1.png \
  -M up="weathered terracotta clay roof tiles in rows" \
  -M side="rough grey fieldstone wall with pale mortar" \
  -M rest="plain dark stone slab" \
  -i "subtle moss near the ground, no strong shadows"
```

What happens:

1. Each (material × UV rotation) runs as one masked, upright model call with a little
   surrounding context, in parallel.
2. The largest pass of each material is painted first. The other passes of that material get it
   as a scale/look reference, so the parts match.
3. **Offset correction.** Models often paint their area shifted by 1–2% (12–20 px on a 1024
   atlas was seen on every call in testing). Each pass measures where the changed texels actually
   sit against the mask and translates the patch back (`shift_corrected` in the call info).
4. The result is clamped to the islands.
5. Clean-up: small unpainted strips (≤5% of an island) are filled by mirroring nearby painted
   texture, tone is blended across seams that were continuous in the original, and island colours
   are bled 16 px into the padding.
6. `check` runs, and islands that fail are repaired with masked edits, up to `--repair-rounds` (2).

Outputs:

- `atlas_v1.png`: the result
- `.raw.png`: the model's unclamped output, which the drift check uses
- `.guide.png`: the outline guide
- `.check.json`: the full report

Exit code 0 means the rules pass, and 2 means errors remain. A single pass is about 1–3 minutes
on Codex. Use `--mode whole` only for single-material atlases whose islands are all upright,
because it can't orient patterns per island.

## 5. Read the check report

`imagegen uv check new.png --original old.png --uv work/uv --raw new.raw.png -M ... [--judge --style ref.png]`
re-runs the checks on any atlas. The AI judge is off by default in both `check` and `retexture`;
leave it off unless the user asks for it. Pass the same `-M` materials, so `coverage` knows which texels
were supposed to be repainted. The meaning of each check and its threshold is in
`references/atlas-rules.md`; read it when a check fails and you need to decide what to do.
In short:

- **Errors** fail the atlas and trigger repairs:
  - `size_match`
  - `coverage`: per island, more than 0.5% of texels left unpainted. That means transparent, flat
    background colour, or still identical to the original atlas. The last case catches a strip of
    old colour, or of a neighbouring material, uncovered by offset paint.
  - `padding_bleed`
  - `layout_registration`: drift on islands whose original had painted features that must stay put
- **Warnings** need your judgement, usually on the renders:
  - `layout_spill`: drift on featureless islands. Overflow past the edge is trimmed by clamping,
    but an *offset* leaves the island's leading edge unpainted. `coverage` and the renders are where
    that shows. Always look at island edges in the renders when this appears.
  - `seam_continuity`
  - `feature_preservation`
  - `format`
  - `uv_layout_sanity`
  - `vision_review`: the atlas-level AI judge (only with `--judge`)

**Don't use `--judge` by default.** It adds 30–60 s per check and is often wrong. If you do use
it, don't act on its island flags without confirming them on renders. Two false
positives came up repeatedly in testing:
- **Rotated islands.** The judge sees correct tile rows running sideways on an island rotated in
  UV space, and calls them wrong.
- **Mixed islands.** An island that is half wall, half roof gets flagged as "roof tiles on a
  wall".

## 6. Verify on the mesh (required)

The atlas checks can't see how the texture sits on the 3D surface. Always render, then **look at
the renders yourself** (read the PNGs) next to the before renders and the reference:

```bash
imagegen uv render scene.blend --object "House" --image work/atlas_v1.png -o work/renders/v1 --opaque
```

Your own look is the default review. Skip the AI render judge unless the user asks for a score or
you can't tell whether something is wrong. It costs about 60 s per call. If you do run it:

```bash
imagegen uv review-renders --after work/renders/v1 --before work/renders/before --style ref.png
```

It reports:

- `style_match` 0–10, and `improved` (compared with the before renders)
- `visible_seams`
- `stretching_or_misalignment`
- `painted_lighting`
- `pattern_scale_or_direction_problems`
- issues and fix suggestions

Treat its output as evidence, not a verdict. Check its claims against the geometry: in testing it
suggested putting roof tiles on the stone gable walls. Look at the island `faces` fractions and
the reference photo before following a material suggestion.

Some suggestions are outside this tool, for example "add normal maps", "re-unwrap" or "use
triplanar mapping". Pass those on to the user as recommendations instead of trying to do them.

Look closely at **island edges**: base of walls, eaves, corners. That's where leftover strips from
offset paint show up, and the checks were blind to them before the coverage fix.

## 7. Fix loop

| what you see | likely cause | do this |
|---|---|---|
| wrong material on a part (render) | selector plan doesn't match the geometry | adjust `-M` selectors/order (check island normals), re-run |
| patterns sideways/stretched on some faces | old export without rotation data, or `--mode whole` | re-export, use material passes |
| one island bad (smeared, off-style, holes) | model failure on that pass | `uv repair new.png --original old.png --uv work/uv -s ref.png --islands 3 --raw new.raw.png -M ... -o v2.png` |
| thin band of old colour / neighbour material along an island edge | paint offset the corrector didn't fully undo | `uv fill-holes new.png --original old.png --uv work/uv --raw new.raw.png -M ... -o v2.png` (mirrors nearby texture); `uv repair` that island if the band is wide |
| tone jump at corners | independent passes on each side of a seam | `uv fix-seams new.png --uv work/uv --original old.png -o v2.png`; pattern continuity across seams can't be fully fixed in 2D (say so) |
| baked shadows/gradients | reference photo lighting | `uv flatten new.png --original old.png --uv work/uv -M ... -o v2.png` on the existing atlas (large-scale gradients only; per-stone shadows stay). Compare the renders before keeping it, since it doesn't always help |
| stone/tile size differs between parts | passes drifted, or UV texel density differs (see sanity) | re-run (anchors match passes); if texel density varies, tell the user |
| thin dark/bright lines at island edges in renders | missing padding | `uv bleed new.png --uv work/uv --padding 16 -o v2.png` |
| looks like a photo of the object, not a texture | whole-atlas call ignored the layout | use material passes; check `coverage`/`layout_registration` |
| `codex exec failed ... stream disconnected` | network blip | retried automatically; re-run if it persists |

Stop after 2–3 iterations and report honestly. Don't loop forever chasing a judge score.

## 8. Deliver

- **Don't overwrite files.** Never overwrite the user's original atlas or .blend without asking.
  Write the new atlas next to it, e.g. `house_atlas_retextured.png`.
- **Hand over a textured .blend.** Offer
  `uv render ... --image new.png --save scene_retextured.blend`, which saves a *copy* with the new
  texture wired in. Alternatively tell them to point the Image Texture node at the new file and
  reload it.
- **Report back:**
  - before/after renders (send the images)
  - the check summary: rules passed, warnings left and why they're acceptable
  - the render-review score
  - any limitations, such as seam pattern continuity or baked lighting inherited from a photo
    reference

## Python API (when scripting)

```python
from imagegen import get_backend, uvtex, blender_bridge
blender_bridge.export_uv("scene.blend", "work/uv", obj="House")
uv = uvtex.UVLayout.load("work/uv")
b = get_backend("codex")
mats = [("up", "terracotta roof tiles"), ("side", "fieldstone wall"), ("rest", "dark stone")]
res = uvtex.retexture("atlas.png", uv, ["ref.png"], backend=b, materials=mats)
rep = uvtex.check(res.image, "atlas.png", uv, raw=res.raw, style="ref.png", backend=b, materials=mats)
if rep["repair_islands"]:
    final, raw = uvtex.repair(res.image, "atlas.png", uv, rep["repair_islands"], ["ref.png"], backend=b, raw=res.raw, materials=mats)
```

The same operations are exposed as MCP tools (`uv_export`, `uv_retexture`, `uv_check`,
`uv_repair`, `uv_render`, `uv_review_renders`) by `imagegen-mcp`.
