import bpy, sys
img = bpy.data.images.load(sys.argv[-1])
w, h = img.size
px = list(img.pixels)
def at(x, y):  # y from top
    i = ((h - 1 - y) * w + x) * 4
    return tuple(round(v, 3) for v in px[i:i+3])
for (x, y) in [(20,20),(700,20),(500,200),(700,390),(450,390),(100,100),(600,100)]:
    print((x, y), at(x, y))
