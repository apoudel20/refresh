# Usage: blender -b -P aux/compare.py -- render.png out.png   (stacks reference above render)
import bpy, sys, numpy as np, os
a = sys.argv[sys.argv.index("--") + 1:]
ref = bpy.data.images.load(os.path.join(os.path.dirname(__file__), "reference.png"))
ren = bpy.data.images.load(a[0])
def arr(img, w, h):
    img.scale(w, h)
    return np.array(img.pixels[:]).reshape(h, w, 4)
w, h = 740, 408
A, B = arr(ref, w, h), arr(ren, w, h)
out = bpy.data.images.new("cmp", w, h * 2)
out.pixels = np.concatenate([B, A], 0).ravel()   # rows bottom-up: render bottom, ref top
out.filepath_raw = a[1]; out.file_format = "PNG"; out.save()
