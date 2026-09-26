# Blender texture-atlas rules and how each is checked

Read this when a check fails or a render looks wrong and you need to decide what to do.
The check names are the ones in `*.check.json` and in `imagegen uv check`.

## Contents
1. Size and aspect are fixed
2. Content stays registered to its UV island
3. Every island is fully painted
4. Padding (bleed) around islands
5. Nothing crosses into empty space or other islands
6. Colour continuity across UV seams
7. Pattern direction follows the surface
8. Consistent texel scale
9. Albedo only: no baked lighting, no text
10. File format and colour space
11. UV-layout problems the texture can't fix
12. What only renders can show

---

## 1. Size and aspect are fixed — `size_match` (error)

UV coordinates are normalised (0–1). A texture with a different aspect ratio stretches on the
mesh. A different resolution with the same aspect is technically fine, but it breaks the padding
and any pixel-exact tooling. The pipeline always writes at the original size. If this check
fails, something outside the pipeline resized the image.

## 2. Content stays registered to its UV island — `layout_registration` (error) / `layout_spill` (warning)

Image models redraw the whole canvas and often shift, rescale or re-crop it by 1–2%. In one test
run a wall pass came back offset 12–20 px on 3 of 3 calls. The damage from an offset:

- **Leading edge.** The texels the paint moved away from keep whatever was there before: old flat
  colour, background, or the neighbouring material, e.g. roof red at the base of a wall. The
  result is a visible band along one edge of the island. Clamping does not fix this. It only
  trims paint that overflowed *out* of the island.
- **Painted features** (eyes, windows, panel lines) move off the geometry they belong to.

What the pipeline does:

- **Prevention.** Every masked pass compares the model's output with its input, finds the
  translation that best lines the changed texels up with the mask, and shifts the patch back
  before compositing. The call info records it as `shift_corrected: [dx, dy]`. Offsets under
  about 3 px are left alone; the hole filler handles strips that thin.
- **Detection, drift.** The check takes each island's true outline from Blender and searches ±2%
  of the texture size for the offset where the model's edges line up best:
  - registered: best offset (0, 0) and prominence well above 1.25 (good results: 5–10)
  - drifted: offset larger than `max(2 px, 0.4% of size)`
  - outline lost: prominence ≈ 1, the model ignored the island shape
  - Drift is an **error** on islands whose original has painted features, and a warning
    (`layout_spill`) elsewhere.
- **Detection, strips.** An uncovered strip is caught directly by `coverage` (§3). When
  `layout_spill` appears, look at that island's edges in the renders.

Use `--raw <atlas>.raw.png` when checking by hand. Without it the check only sees the clamped
result, which hides drift.

## 3. Every island is fully painted — `coverage` (error)

Measured **per island**. The error fires when any island has more than 0.5% of its texels unpainted.
An overall average would hide a 2% strip on one island behind many clean ones, and that happened.

A texel counts as unpainted if it is:
- transparent, or
- flat empty-background colour, or
- still locally identical to the original atlas.

The last case catches strips of old colour or of a neighbouring material left by offset paint.
Dark or grey *textured* materials that happen to resemble the background are not holes. With
`-M` materials given, only texels assigned a material are expected to change.

What to do about it:
- Islands at up to 5% unpainted get their strips filled automatically during `retexture`, by
  mirroring nearby painted texels across the strip's edge (`uv fill-holes` does the same on
  demand).
- Anything larger goes to `repair`, because mirroring can't invent that much texture.

## 4. Padding (bleed) around islands — `padding_bleed` (error)

Mip-mapping and bilinear filtering sample texels *outside* island edges. If those are background
colour, the model shows thin dark or bright lines along every UV seam, which get worse at a
distance. Blender's own bake uses a 16 px margin by default. The pipeline extends each island's
edge colours outward by `--padding` px, taking the nearest island texel. The check requires ≥90%
of the inner half of that margin to carry island colour. Fix with `imagegen uv bleed`.

Pick a padding up to about half the smallest gap between islands. Larger values are harmless:
the bleed stops where the nearest texel belongs to another island.

## 5. Nothing crosses into empty space or other islands — `outside_islands_untouched` (warning)

Texels farther from any island than the padding should be byte-identical to the original.
Clamping guarantees this, so a failure means the atlas was edited outside this pipeline. That
usually matters only when the atlas is shared with other objects, whose islands live in that
"empty" space as far as this object is concerned. **If the image is shared by several objects,
export UVs for all of them** and merge the island maps. Otherwise ask the user.

## 6. Colour continuity across UV seams — `seam_continuity` (warning)

A UV seam is a mesh edge whose two sides sit in different places in the atlas. On the model they
touch. The check samples both sides of every seam (blurred, so it compares tone rather than
pattern). It counts seam length that was continuous in the original atlas but now jumps by more
than about 30/255.

- `retexture` already blends tone across those seams, and so does `uv fix-seams --original`.
  Only seams that were continuous in the original are touched, so intended material boundaries
  (roof edge against wall) stay crisp.
- **Pattern continuity is a real limit.** Individual stones or planks can't be made to continue
  across a seam in 2D. Corners will show the pattern restart. Tell the user.
  - The fixes are outside this tool: fewer seams (re-unwrap), triplanar or box projection in the
    shader, or texture painting in Blender.

## 7. Pattern direction follows the surface — rendering + `up_rotation`

Islands are often rotated (90°/180°/270°) or mirrored in UV space. A pattern painted "upright" in
2D then runs sideways or upside-down on the model: vertical brick courses, sideways tile rows,
roofs whose tiles point up. The export records where world-up points for every texel. Horizontal
faces use world +Y as "up".

Material passes are split by that rotation and painted on an atlas turned upright for each pass,
using lossless 90° rotations. `--mode whole` can't do this and warns when rotations are mixed.
Mirrored islands (`flipped_faces` > 0) are not un-mirrored, so directional details such as text,
arrows or asymmetric patterns can appear flipped. Mention it if the user cares.

## 8. Consistent texel scale — `uv_layout_sanity.texel_density_cv`, renders

A pattern painted at one pixel scale looks larger on islands with lower texel density (fewer
pixels per metre). `texel_density_cv` > 0.25 means the unwrap's density varies noticeably. The
pipeline rescales anchor references by the density ratio between passes, but it doesn't resample
islands. Mention it as a UV issue if renders show mismatched scale.

Separate model passes can also drift in scale. The first (largest) pass of each material is used
as a reference for the others to counter that.

## 9. Albedo only: no baked lighting, no text — `vision_review` (warning), renders

A base-colour map should hold surface colour only. Shading, cast shadows, highlights and ambient
occlusion painted into it fight the scene lighting, and look wrong once the object rotates.
Photo references carry lighting, so expect some.

- `uv flatten` (or `retexture --flatten-lighting`) removes large-scale gradients per material group
  while keeping fine detail. Compare renders before keeping it: in one test it gave no visible
  benefit.
- Small per-stone shadows stay. Report them rather than chasing them.
- Text, numbers and watermarks are never acceptable. Repair the island.

## 10. File format and colour space — `format` (warning)

- Keep 8-bit RGB PNG.
- Don't add an alpha channel unless the material uses one: Blender's Principled BSDF alpha, or
  blend modes, can turn it into transparency.
- Base colour images must use the sRGB colour space (reported in `uv_info.json` → `textures`).
- Power-of-two sizes are preferred by game engines but not required by Blender. Informational only.

## 11. UV-layout problems the texture can't fix — `uv_layout_sanity` (warning)

| field | meaning | consequence |
|---|---|---|
| `overlap_texels` | islands overlap in UV | both surfaces get identical paint (often intentional mirroring) |
| `faces_outside_0_1` | UVs beyond the 0–1 square | the texture tiles there; that area isn't in this atlas |
| `flipped_faces` | mirrored UV faces | directional details appear mirrored |
| `texel_density_cv` | spread of pixels-per-metre across islands | inconsistent pattern scale |
| `udim` | tiled (UDIM) image | handle each tile as its own atlas |

Report these to the user; fixing them means re-unwrapping in Blender.

## 12. What only renders can show

Some problems only exist on the 3D surface:

- wrong material on a part
- sideways patterns
- seams at corners
- stretching where the UVs are distorted
- baked lighting fighting the scene

`imagegen uv render` + `review-renders`, plus your own look at the PNGs, are the final check.
A passing `check` with bad renders is still a failed retexture.
