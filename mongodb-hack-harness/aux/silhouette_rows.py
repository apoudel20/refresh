"""Per-row silhouette extents (dog pixels) for reference vs render, plus key landmarks.
Usage: blender -b --factory-startup -P aux/silhouette_rows.py -- render.png"""
import bpy, sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 740, 408
def load(p):
    im = bpy.data.images.load(p)
    if tuple(im.size) != (W, H): im.scale(W, H)
    return np.array(im.pixels[:], np.float32).reshape(H, W, 4)[::-1, :, :3]
def mask(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return ~((b - r > 0.22) & (g > 0.45) & (b > 0.6))
ref, ren = load(os.path.join(HERE, "reference.png")), load(sys.argv[-1])
mr, mn = mask(ref), mask(ren)
def runs(row):
    xs = np.nonzero(row)[0]
    if not len(xs): return []
    br = np.nonzero(np.diff(xs) > 3)[0]
    st = np.r_[xs[0], xs[br + 1]]; en = np.r_[xs[br], xs[-1]]
    return [(int(a), int(b)) for a, b in zip(st, en) if b - a > 2]
print("row   ref_runs                         render_runs")
for y in range(0, H, 12):
    print(f"{y:3d}   {str(runs(mr[y])):32s} {runs(mn[y])}")
top = lambda m: int(np.nonzero(m.any(1))[0][0])
print("top row: ref", top(mr), "render", top(mn))
