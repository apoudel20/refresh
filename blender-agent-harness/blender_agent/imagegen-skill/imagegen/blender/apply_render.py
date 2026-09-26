"""Put a texture on an object and render check views, from inside Blender.

Run (background; the .blend is never overwritten — use --save to write a copy):
    blender -b scene.blend --factory-startup --python imagegen/blender/apply_render.py -- \
        --out-dir renders/after --image new_atlas.png [--object NAME] [--replace old.png]
        [--views front,right,back,iso] [--res 768] [--engine eevee|workbench|cycles]
        [--samples 32] [--save copy.blend]

Without --image the object renders with its current texture (use for "before" shots).
--replace limits the swap to Image Texture nodes whose image file matches that path
(default: every Image Texture node feeding a Base Color / Color input on the object).
If the object has no material, one is created (Principled BSDF <- Image Texture <- UV).

Cameras are placed around the object's world bounding box; lighting is a neutral grey
world + soft sun so painted-in lighting in the texture is easy to spot.
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

VIEWS = {
    "front": Vector((0, -1, 0.15)),
    "back": Vector((0, 1, 0.15)),
    "left": Vector((-1, 0, 0.15)),
    "right": Vector((1, 0, 0.15)),
    "top": Vector((0, -0.01, 1)),
    "bottom": Vector((0, 0.01, -1)),
    "iso": Vector((1, -1, 0.8)),
    "iso_back": Vector((-1, 1, 0.8)),
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    opts = {"out_dir": None, "image": None, "object": None, "replace": None, "views": "front,right,back,iso",
            "res": "768", "engine": "eevee", "samples": "32", "save": None, "transparent": "1"}  # fmt: skip
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-").replace("-", "_")
        if key in opts and i + 1 < len(argv):
            opts[key] = argv[i + 1]
            i += 2
        else:
            raise SystemExit(f"apply_render: unknown argument {argv[i]!r}")
    if not opts["out_dir"]:
        raise SystemExit("apply_render: --out-dir is required")
    return opts


def pick_object(name):
    if name:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise SystemExit(f"apply_render: no mesh object {name!r}")
        return obj
    active = bpy.context.view_layer.objects.active
    if active is not None and active.type == "MESH":
        return active
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.data.uv_layers]
    if len(meshes) == 1:
        return meshes[0]
    raise SystemExit("apply_render: pass --object; meshes: " + str([o.name for o in meshes]))


def same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


def swap_texture(obj, image_path, replace):
    img = bpy.data.images.load(os.path.abspath(image_path), check_existing=False)
    img.colorspace_settings.name = "sRGB"
    swapped = []
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            if replace:
                cur = bpy.path.abspath(node.image.filepath) if node.image.filepath else ""
                if not cur or not same_file(cur, replace):
                    continue
            else:
                feeds_color = any(
                    "Color" in l.to_socket.name for o in node.outputs for l in o.links
                )
                if not feeds_color:
                    continue
            node.image = img
            swapped.append(f"{mat.name}/{node.name}")
    has_material = any(s.material is not None for s in obj.material_slots)
    if not swapped and not has_material:
        mat = bpy.data.materials.new(f"{obj.name}_retexture")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = obj.data.uv_layers.active.name if obj.data.uv_layers.active else ""
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        swapped.append(f"{mat.name}/{tex.name} (new material)")
    if not swapped:
        raise SystemExit("apply_render: found no Image Texture node to replace (try --replace <current image path>)")
    return swapped


def set_engine(scene, engine, samples):
    items = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    wanted = {
        "eevee": ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
        "workbench": ["BLENDER_WORKBENCH"],
        "cycles": ["CYCLES"],
    }[engine]
    ident = next((e for e in wanted if e in items), "BLENDER_WORKBENCH")
    scene.render.engine = ident
    if ident == "BLENDER_WORKBENCH":
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "TEXTURE"
    elif ident == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.device = "CPU"
    else:
        try:
            scene.eevee.taa_render_samples = samples
        except AttributeError:
            pass
    return ident


def setup_lighting(scene):
    world = bpy.data.worlds.new("retexture_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.6, 0.6, 0.6, 1)
    bg.inputs["Strength"].default_value = 0.8
    scene.world = world
    sun_data = bpy.data.lights.new("retexture_sun", "SUN")
    sun_data.energy = 2.0
    sun_data.angle = math.radians(20)
    sun = bpy.data.objects.new("retexture_sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene.collection.objects.link(sun)


def world_bounds(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return (lo + hi) / 2, max((hi - lo).length / 2, 1e-3)


def main():
    opts = parse_args()
    scene = bpy.context.scene
    obj = pick_object(opts["object"])
    swapped = swap_texture(obj, opts["image"], opts["replace"]) if opts["image"] else []

    # Render only the target object so other scene clutter can't hide problems.
    for o in scene.objects:
        if o.type in ("MESH", "CURVE", "SURFACE", "META", "FONT", "GPENCIL", "GREASEPENCIL", "VOLUME") and o != obj:
            o.hide_render = True
    for o in [o for o in scene.objects if o.type == "LIGHT"]:
        o.hide_render = True
    setup_lighting(scene)

    engine = set_engine(scene, opts["engine"], int(opts["samples"]))
    res = int(opts["res"])
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = opts["transparent"] != "0"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    center, radius = world_bounds(obj)
    cam_data = bpy.data.cameras.new("retexture_cam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("retexture_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    fov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
    dist = radius / math.sin(fov / 2) * 1.05
    cam_data.clip_end = dist + radius * 4

    os.makedirs(opts["out_dir"], exist_ok=True)
    written = []
    for view in [v.strip() for v in opts["views"].split(",") if v.strip()]:
        if view not in VIEWS:
            raise SystemExit(f"apply_render: unknown view {view!r}; choose from {sorted(VIEWS)}")
        d = VIEWS[view].normalized()
        cam.location = center + d * dist
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        path = os.path.join(opts["out_dir"], f"{view}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        written.append(path)

    if opts["save"]:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(opts["save"]), copy=True)
    print("APPLY_RENDER_OK " + json.dumps({"renders": written, "engine": engine, "swapped": swapped, "object": obj.name}))


main()
