"""Save the black/tan/white classification of the reference next to it (debug view)."""
import bpy, sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build_dog as bd
alb, valid = bd.reference_albedo(os.path.join(HERE, "reference.png"))
vis = np.where(valid[..., None], np.clip(alb / 0.85, 0, 1) ** (1 / 2.2), [0.2, 0.4, 0.6])
im = bpy.data.images.load(os.path.join(HERE, "reference.png")); w, h = im.size
ref = np.array(im.pixels[:], np.float32).reshape(h, w, 4)[::-1, :, :3]
c = np.concatenate([ref, vis], 1)
out = bpy.data.images.new("a", c.shape[1], c.shape[0])
out.pixels = np.concatenate([c[::-1], np.ones(c.shape[:2] + (1,))], 2).ravel()
out.filepath_raw = os.path.join(HERE, "renders", "albedo_class.png"); out.file_format = "PNG"; out.save()
