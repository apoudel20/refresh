"""Side-by-side zoomed crop of reference and render with a shared pixel grid.
Usage: blender -b -P aux/pair_crop.py -- render.png out.png x0 y0 x1 y1 scale step"""
import bpy, sys, os, numpy as np
a = sys.argv[sys.argv.index("--") + 1:]
HERE = os.path.dirname(os.path.abspath(__file__))
def load(p):
    im = bpy.data.images.load(p)
    if tuple(im.size) != (740, 408): im.scale(740, 408)
    return np.array(im.pixels[:], np.float32).reshape(408, 740, 4)[::-1]
x0, y0, x1, y1, sc, step = map(int, a[2:8])
outs = []
for px in (load(os.path.join(HERE, "reference.png")), load(a[0])):
    c = px[y0:y1, x0:x1].repeat(sc, 0).repeat(sc, 1).copy()
    for gx in range((x0 // step + 1) * step, x1, step):
        c[:, (gx - x0) * sc] = [1, 0, 0, 1] if gx % (step * 5) == 0 else [1, 1, 0, 1]
    for gy in range((y0 // step + 1) * step, y1, step):
        c[(gy - y0) * sc, :] = [1, 0, 0, 1] if gy % (step * 5) == 0 else [1, 1, 0, 1]
    outs.append(c)
c = np.concatenate([outs[0], np.ones((outs[0].shape[0], 6, 4)), outs[1]], 1)
out = bpy.data.images.new("o", c.shape[1], c.shape[0]); out.pixels = c[::-1].ravel()
out.filepath_raw = a[1]; out.file_format = "PNG"; out.save()
