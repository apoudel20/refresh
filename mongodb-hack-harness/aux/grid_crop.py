"""Draw a labelled pixel grid on an image crop (for measuring landmarks).
Usage: blender -b -P aux/grid_crop.py -- in.png out.png x0 y0 x1 y1 scale step"""
import bpy, sys, numpy as np
a = sys.argv[sys.argv.index("--") + 1:]
im = bpy.data.images.load(a[0]); W, H = im.size
px = np.array(im.pixels[:], np.float32).reshape(H, W, 4)[::-1]
x0, y0, x1, y1, sc, step = map(int, a[2:8])
c = px[y0:y1, x0:x1].repeat(sc, 0).repeat(sc, 1)
for gx in range((x0 // step + 1) * step, x1, step):
    X = (gx - x0) * sc; c[:, X] = [1, 0, 0, 1] if gx % (step * 5) == 0 else [1, 1, 0, 1]
for gy in range((y0 // step + 1) * step, y1, step):
    Y = (gy - y0) * sc; c[Y, :] = [1, 0, 0, 1] if gy % (step * 5) == 0 else [1, 1, 0, 1]
out = bpy.data.images.new("o", c.shape[1], c.shape[0]); out.pixels = c[::-1].ravel()
out.filepath_raw = a[1]; out.file_format = "PNG"; out.save()
