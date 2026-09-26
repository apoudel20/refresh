# Usage: blender -b -P aux/grid.py -- out.png a.png b.png c.png d.png   (2x2 grid)
import bpy, sys, numpy as np
a = sys.argv[sys.argv.index("--") + 1:]
imgs = [bpy.data.images.load(f) for f in a[1:]]
w, h = imgs[0].size
arr = [np.array(i.pixels[:]).reshape(h, w, 4) for i in imgs]
while len(arr) < 4: arr.append(np.ones_like(arr[0]))
top = np.concatenate([arr[0], arr[1]], 1); bot = np.concatenate([arr[2], arr[3]], 1)
out = bpy.data.images.new("g", w * 2, h * 2)
out.pixels = np.concatenate([bot, top], 0).ravel()
out.filepath_raw = a[0]; out.file_format = "PNG"; out.save()
