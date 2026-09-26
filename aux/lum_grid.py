"""Print per-patch mean luminance of reference and render (dog pixels only) as two grids + difference.
Usage: blender -b --factory-startup -P aux/lum_grid.py -- render.png [cols rows]"""
import bpy, sys, os, numpy as np
a = sys.argv[sys.argv.index("--") + 1:]
HERE = os.path.dirname(os.path.abspath(__file__))
C = int(a[1]) if len(a) > 1 else 20; Rw = int(a[2]) if len(a) > 2 else 11
def load(p):
    im = bpy.data.images.load(p)
    if tuple(im.size) != (740, 408): im.scale(740, 408)
    return np.array(im.pixels[:], np.float32).reshape(408, 740, 4)[::-1, :, :3]
def mask(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return ~((b - r > 0.22) & (g > 0.45) & (b > 0.6))
ref, ren = load(os.path.join(HERE, "reference.png")), load(a[0])
L = lambda im: im @ np.array([0.299, 0.587, 0.114])
lr, ln, mr, mn = L(ref), L(ren), mask(ref), mask(ren)
ph, pw = 408 / Rw, 740 / C
print("diff (render - ref) mean luminance x100 on shared dog pixels; '.' = no overlap")
print("     " + "".join(f"{i:5d}" for i in range(C)))
for j in range(Rw):
    row = []
    for i in range(C):
        y0, y1, x0, x1 = int(j * ph), int((j + 1) * ph), int(i * pw), int((i + 1) * pw)
        m = mr[y0:y1, x0:x1] & mn[y0:y1, x0:x1]
        row.append(f"{int(round(100 * (ln[y0:y1, x0:x1][m].mean() - lr[y0:y1, x0:x1][m].mean()))):5d}" if m.sum() > 30 else "    .")
    print(f"{j:3d}  " + "".join(row))
