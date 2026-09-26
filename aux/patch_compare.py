"""Patch-by-patch comparison of a render against aux/reference.png.

Usage: blender -b --factory-startup -P aux/patch_compare.py -- render.png out_prefix [cols rows]

Writes <out_prefix>_heat.png (reference | render | per-patch error heat with grid, labelled)
and prints per-patch scores, worst first. Scores per patch:
  sil  = fraction of pixels where the dog/background classification disagrees (shape error)
  col  = mean colour distance over pixels that are dog in both images (appearance error)
  lum  = mean luminance difference (render - reference) on shared dog pixels (+ = render too bright)
"""
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[sys.argv.index("--") + 1:]
render_path, prefix = a[0], a[1]
COLS = int(a[2]) if len(a) > 2 else 20
ROWS = int(a[3]) if len(a) > 3 else 11


def load(path, w, h):
    img = bpy.data.images.load(path)
    if tuple(img.size) != (w, h):
        img.scale(w, h)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]  # top row first
    return px


W, H = 740, 408
ref = load(os.path.join(HERE, "reference.png"), W, H)
ren = load(render_path, W, H)


def dog_mask(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    bg = (b - r > 0.22) & (g > 0.45) & (b > 0.6)
    return ~bg


mr, mn = dog_mask(ref), dog_mask(ren)
lum = lambda im: im @ np.array([0.299, 0.587, 0.114], np.float32)
ph, pw = H / ROWS, W / COLS
rows = []
heat = np.zeros((ROWS, COLS))
for j in range(ROWS):
    for i in range(COLS):
        y0, y1, x0, x1 = int(j * ph), int((j + 1) * ph), int(i * pw), int((i + 1) * pw)
        a_, b_ = mr[y0:y1, x0:x1], mn[y0:y1, x0:x1]
        if not (a_.any() or b_.any()):
            continue
        sil = float((a_ != b_).mean())
        both = a_ & b_
        if both.sum() > 20:
            d = np.linalg.norm(ref[y0:y1, x0:x1][both] - ren[y0:y1, x0:x1][both], axis=1)
            col = float(d.mean())
            dl = float((lum(ren[y0:y1, x0:x1])[both] - lum(ref[y0:y1, x0:x1])[both]).mean())
            rc = ref[y0:y1, x0:x1][both].mean(0)
            nc = ren[y0:y1, x0:x1][both].mean(0)
        else:
            col, dl, rc, nc = 0.0, 0.0, np.zeros(3), np.zeros(3)
        score = sil + 0.6 * col
        heat[j, i] = score
        rows.append((score, j, i, sil, col, dl, rc, nc))

rows.sort(key=lambda r: -r[0])
print(f"PATCH GRID {COLS}x{ROWS} ({pw:.0f}x{ph:.0f}px). mean score {np.mean([r[0] for r in rows]):.3f}  "
      f"silhouette IoU {((mr & mn).sum() / max((mr | mn).sum(), 1)):.3f}")
print(" rank  patch(r,c)  px(x,y)     score   sil    col    dlum   ref_rgb            render_rgb")
for k, (score, j, i, sil, col, dl, rc, nc) in enumerate(rows[:int(os.environ.get("TOPN", 25))]):
    print(f" {k + 1:4d}  ({j:2d},{i:2d})    ({int((i + .5) * pw):3d},{int((j + .5) * ph):3d})  {score:.3f}  {sil:.3f}  "
          f"{col:.3f}  {dl:+.3f}  {np.round(rc, 2)}  {np.round(nc, 2)}")

# heat visual: ref | render | heat overlay on render with grid
hv = np.zeros((H, W, 3), np.float32)
mx = max(heat.max(), 1e-6)
for j in range(ROWS):
    for i in range(COLS):
        y0, y1, x0, x1 = int(j * ph), int((j + 1) * ph), int(i * pw), int((i + 1) * pw)
        t = heat[j, i] / mx
        hv[y0:y1, x0:x1] = ren[y0:y1, x0:x1] * 0.45 + np.array([t, 0.15 * (1 - t), 0.0]) * 0.55
        hv[y0, x0:x1] = hv[y0:y1, x0] = 1.0
# silhouette disagreement in magenta
dis = mr != mn
hv[dis] = hv[dis] * 0.3 + np.array([1, 0, 1]) * 0.7
out = np.concatenate([ref, ren, hv], 1)
img = bpy.data.images.new("heat", out.shape[1], out.shape[0])
rgba = np.concatenate([out[::-1], np.ones(out.shape[:2] + (1,), np.float32)], 2)
img.pixels = rgba.ravel()
img.filepath_raw = prefix + "_heat.png"
img.file_format = "PNG"
img.save()
