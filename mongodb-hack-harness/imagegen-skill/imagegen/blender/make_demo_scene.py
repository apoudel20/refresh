"""Build a small UV-unwrapped test object with a placeholder texture, for trying the retexture flow.

    blender -b --factory-startup --python imagegen/blender/make_demo_scene.py -- --out-dir demo [--size 1024]

Writes demo/house.blend (object "House": walls + gable roof + door, Smart UV Project with margin)
whose material samples demo/house_atlas.png. The PNG starts as a flat grey placeholder;
`imagegen uv placeholder` repaints it with flat per-island colours (the "not good enough" atlas).
"""

import json
import math
import os
import sys

import bmesh
import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    opts = {"out_dir": "demo", "size": "1024"}
    for i in range(0, len(argv) - 1, 2):
        opts[argv[i].lstrip("-").replace("-", "_")] = argv[i + 1]
    return opts


def main():
    opts = parse_args()
    out = os.path.abspath(opts["out_dir"])
    os.makedirs(out, exist_ok=True)
    size = int(opts["size"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("House")
    bm = bmesh.new()
    # walls: 4 x 3 footprint, 2.4 high
    w, d, h, rh = 2.0, 1.5, 2.4, 1.3
    base = [bm.verts.new(v) for v in [(-w, -d, 0), (w, -d, 0), (w, d, 0), (-w, d, 0)]]
    top = [bm.verts.new(v) for v in [(-w, -d, h), (w, -d, h), (w, d, h), (-w, d, h)]]
    ridge = [bm.verts.new((-w, 0, h + rh)), bm.verts.new((w, 0, h + rh))]
    for i in range(4):
        bm.faces.new([base[i], base[(i + 1) % 4], top[(i + 1) % 4], top[i]])
    bm.faces.new(list(reversed(base)))  # floor
    bm.faces.new([top[0], top[1], ridge[1], ridge[0]])  # front roof slope
    bm.faces.new([top[2], top[3], ridge[0], ridge[1]])  # back roof slope
    bm.faces.new([top[1], top[2], ridge[1]])  # right gable
    bm.faces.new([top[3], top[0], ridge[0]])  # left gable
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=2, use_grid_fill=True)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("House", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.03)
    bpy.ops.object.mode_set(mode="OBJECT")

    atlas_path = os.path.join(out, "house_atlas.png")
    img = bpy.data.images.new("house_atlas", size, size, alpha=False)
    img.generated_color = (0.5, 0.5, 0.5, 1)
    img.filepath_raw = atlas_path
    img.file_format = "PNG"
    img.save()

    mat = bpy.data.materials.new("HouseMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.name = "Atlas"
    tex.image = bpy.data.images.load(atlas_path)
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "house.blend"))
    print("DEMO_SCENE_OK " + json.dumps({"blend": os.path.join(out, "house.blend"), "atlas": atlas_path, "object": "House"}))


main()
