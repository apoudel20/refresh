# What each step measures

Both images are first cropped to the object (alpha channel, or a background-removal
model when there is no alpha) and put on the same grey background, so framing and
backdrop don't count. Every step then scores the aligned pair from 0 to 1.

| # | step | measures | score | typical fix when low |
|---|---|---|---|---|
| 1 | pixel | PSNR, SSIM, LPIPS perceptual distance on the object | 0.5 SSIM + 0.5 (1 − LPIPS) | overall look; usually improves with everything else |
| 2 | depth | Depth Anything V2 depth maps, rank correlation where both objects overlap | Spearman correlation | 3D form: bulges, recesses, what sticks out |
| 3 | normals | Marigold surface normals, angle between them | share of pixels within 22.5° | local shape: curvature, creases, facial planes |
| 4 | silhouette | overlap of the two object masks, outline distance | IoU | outline and proportions |
| 5 | edges | Canny edges matched within 0.75% of the diagonal | edge F1 | interior structure: eyes, mouth, markings, part borders |
| 6 | embedding | cosine of Voyage multimodal image embeddings (OpenRouter) | cosine | "does it read as the same thing" |
| 7 | color | shared 8-color palette, earth mover's distance in CIELAB | exp(−EMD / 20) | base colors, markings, how much of each color |
| 8 | judge | vision LLM with a rubric: shape, proportions, parts, color, materials | mean criterion, rescaled | read its notes and top fixes |

Composite weights: silhouette 0.20, depth 0.15, normals 0.15, judge 0.15, pixel 0.10,
edges 0.10, embedding 0.10, color 0.05. They are uncalibrated starting points; override
with `--weights '{"judge": 0.3}'`.

# Token layout (8 tokens x 32 slots)

Tokens are in step order: pixel, depth, normals, silhouette, edges, embedding, color,
judge. In every token, slot 0 is the step's score and slot 31 is 1 when the step ran.
Grids are 4 x 4 over the reference's aligned frame, row by row from the top left
(`r0c0` .. `r3c3`), so a cell means the same region of the object in every run against
the same reference.

| token | slots after the score | grid (16 slots) |
|---|---|---|
| pixel | ssim_fg, lpips_fg, ssim, lpips, psnr_fg | LPIPS per cell: where it looks different |
| depth | spearman, pearson, ssi_mae, aligned_nrmse, discontinuity_f1, overlap | + = render nearer the camera than the reference there, − = farther |
| normals | mean and median agreement, within 11.25°, within 30° | mean angle error per cell / 90 |
| silhouette | dice, boundary_f1, chamfer, hd95, area log-ratio, aspect log-ratio | + = extra area in the render, − = missing area |
| edges | precision, recall, interior_f1, chamfer, density log-ratio | + = edges only in the render, − = edges only in the reference |
| embedding | vectors_present, then 24 slots: fixed random projection of the embedding difference | none |
| color | palette EMD, EMD ignoring lightness, ΔE2000, Δ lightness, Δa, Δb, Δ contrast, then 12 bins (4 lightness levels x neutral/warm/cool), + = more of that color in the render | none |
| judge | overall, shape, proportions, parts, color, materials (each /10), missing and extra part counts (/5), text_present, then 16 slots: projection of a text embedding of the critic's notes | none |

`render-eval vectors --export v.json` writes the full list of 256 slot names. 191 slots
carry information; the rest are padding that is always zero.
