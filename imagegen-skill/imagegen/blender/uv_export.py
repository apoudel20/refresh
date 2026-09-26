"""Export a mesh's UV layout as data, from inside Blender.

Run (background, never saves the .blend):
    blender -b scene.blend --factory-startup --python imagegen/blender/uv_export.py -- \
        --out-dir uvdir [--object NAME] [--uv-map NAME] [--size 2048 | --size 2048x1024]

Writes to --out-dir:
    uv_data.npz   islands: int32 [H, W] island id per texel (0 = empty), image orientation
                  (row 0 = top, i.e. UV v=1); overlap: uint8 [H, W] texels covered by >1 island;
                  normals: float16 [H, W, 3] world-space face normal per texel (which way it faces);
                  up_angle: float16 [H, W] where world "up" points in the image at each texel, in degrees
                  clockwise from image-up (NaN = empty). Horizontal faces use world +Y as "up".
    uv_info.json  object/mesh/uv map, texture images found on the materials, per-island
                  bbox/area/texel density, UV seams as pixel segments, sanity counts

Only bpy, bmesh and numpy (bundled with Blender) are used. Pixel coordinates are x right,
y down, pixel centres at +0.5 — the same as PIL/numpy on the saved texture.
"""

import json
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from bpy_extras import bmesh_utils


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    opts = {"out_dir": None, "object": None, "uv_map": None, "size": None}
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-").replace("-", "_")
        if key in opts and i + 1 < len(argv):
            opts[key] = argv[i + 1]
            i += 2
        else:
            raise SystemExit(f"uv_export: unknown argument {argv[i]!r}")
    if not opts["out_dir"]:
        raise SystemExit("uv_export: --out-dir is required")
    return opts


def pick_object(name):
    if name:
        obj = bpy.data.objects.get(name)
        if obj is None:
            meshes = [o.name for o in bpy.data.objects if o.type == "MESH"]
            raise SystemExit(f"uv_export: no object {name!r}. Mesh objects: {meshes}")
        if obj.type != "MESH":
            raise SystemExit(f"uv_export: {name!r} is a {obj.type}, not a MESH")
        return obj
    active = bpy.context.view_layer.objects.active
    if active is not None and active.type == "MESH" and active.data.uv_layers:
        return active
    cands = [o for o in bpy.data.objects if o.type == "MESH" and o.data.uv_layers]
    if len(cands) == 1:
        return cands[0]
    raise SystemExit(
        "uv_export: pass --object; UV-mapped meshes: " + str([o.name for o in cands])
    )


def texture_images(obj):
    """Image Texture nodes on the object's materials (what we'd be replacing)."""
    found = []
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            img = node.image
            uv_map = None
            vec = node.inputs.get("Vector")
            if vec and vec.is_linked and vec.links[0].from_node.type == "UVMAP":
                uv_map = vec.links[0].from_node.uv_map or None
            targets = sorted({f"{l.to_node.name}.{l.to_socket.name}" for o in node.outputs for l in o.links})
            found.append({
                "material": mat.name,
                "node": node.name,
                "image": img.name,
                "filepath": bpy.path.abspath(img.filepath) if img.filepath else None,
                "packed": img.packed_file is not None,
                "size": list(img.size),
                "colorspace": img.colorspace_settings.name,
                "source": img.source,  # FILE / TILED (UDIM) / GENERATED ...
                "uv_map": uv_map,
                "feeds": targets,
            })
    return found


def raster_triangles(tris, tri_ids, tri_normals, tri_up, W, H):
    """Rasterise UV triangles (pixel coords, y down) into island-id/normal/up-angle maps, counting overlaps."""
    ids = np.zeros((H, W), np.int32)
    overlap = np.zeros((H, W), np.uint8)
    normals = np.zeros((H, W, 3), np.float16)
    up_angle = np.full((H, W), np.nan, np.float16)
    for (a, b, c), k, nrm, up in zip(tris, tri_ids, tri_normals, tri_up):
        x0 = max(int(math.floor(min(a[0], b[0], c[0]))), 0)
        x1 = min(int(math.ceil(max(a[0], b[0], c[0]))), W)
        y0 = max(int(math.floor(min(a[1], b[1], c[1]))), 0)
        y1 = min(int(math.ceil(max(a[1], b[1], c[1]))), H)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1) + 0.5
        ys = np.arange(y0, y1) + 0.5
        X, Y = np.meshgrid(xs, ys)

        def edge(p, q):
            return (q[0] - p[0]) * (Y - p[1]) - (q[1] - p[1]) * (X - p[0])

        w0, w1, w2 = edge(b, c), edge(c, a), edge(a, b)
        eps = -1e-6
        inside = ((w0 >= eps) & (w1 >= eps) & (w2 >= eps)) | ((w0 <= -eps) & (w1 <= -eps) & (w2 <= -eps))
        if not inside.any():
            # Sub-pixel triangle: mark the texel under its centroid so thin islands aren't lost.
            cx, cy = int((a[0] + b[0] + c[0]) / 3), int((a[1] + b[1] + c[1]) / 3)
            if 0 <= cx < W and 0 <= cy < H:
                if ids[cy, cx] not in (0, k):
                    overlap[cy, cx] = 1
                ids[cy, cx] = k
                normals[cy, cx] = nrm
                up_angle[cy, cx] = up
            continue
        sub = ids[y0:y1, x0:x1]
        clash = inside & (sub != 0) & (sub != k)
        overlap[y0:y1, x0:x1][clash] = 1
        sub[inside] = k
        normals[y0:y1, x0:x1][inside] = nrm
        up_angle[y0:y1, x0:x1][inside] = up
    return ids, overlap, normals, up_angle


def face_up_angle(f, uv):
    """Where world-up (projected onto the face) points in image space, degrees clockwise from image-up.

    Solves the face's UV->3D Jacobian J (3x2) and maps the projected up vector back to UV with its
    pseudo-inverse. Image y runs down, so a UV direction (du, dv) is image direction (du, -dv).
    """
    from mathutils import Vector

    n = f.normal
    ref = Vector((0, 0, 1)) if abs(n.z) < 0.95 else Vector((0, 1, 0))
    up = ref - n * ref.dot(n)
    if up.length < 1e-8:
        return float("nan")
    loops = list(f.loops)
    acc = np.zeros(2)
    for i in range(1, len(loops) - 1):
        p0, p1, p2 = (loops[0].vert.co, loops[i].vert.co, loops[i + 1].vert.co)
        t0, t1, t2 = (loops[0][uv].uv, loops[i][uv].uv, loops[i + 1][uv].uv)
        J = np.array([list(p1 - p0), list(p2 - p0)]).T  # 3x2: d3D per (edge1, edge2)
        T = np.array([[t1.x - t0.x, t2.x - t0.x], [t1.y - t0.y, t2.y - t0.y]])  # 2x2
        if abs(np.linalg.det(T)) < 1e-12:
            continue
        M = J @ np.linalg.inv(T)  # 3x2: d3D per dUV
        d = np.linalg.pinv(M) @ np.array(list(up))
        area = np.linalg.norm(np.cross(J[:, 0], J[:, 1]))
        if np.linalg.norm(d) > 0:
            acc += d / np.linalg.norm(d) * area
    if np.linalg.norm(acc) < 1e-12:
        return float("nan")
    du, dv = acc
    return float(np.degrees(np.arctan2(du, dv)) % 360)


def main():
    opts = parse_args()
    obj = pick_object(opts["object"])
    textures = texture_images(obj)

    if opts["size"]:
        s = opts["size"].lower().split("x")
        W, H = (int(s[0]), int(s[0])) if len(s) == 1 else (int(s[0]), int(s[1]))
    else:
        sized = [t for t in textures if t["size"][0] > 0]
        W, H = sized[0]["size"] if sized else (1024, 1024)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    uv_names = [l.name for l in mesh.uv_layers]
    uv_name = opts["uv_map"] or (next((l.name for l in mesh.uv_layers if l.active_render), None) or uv_names[0])
    if uv_name not in uv_names:
        raise SystemExit(f"uv_export: no UV map {uv_name!r}; maps: {uv_names}")

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.transform(obj.matrix_world)  # world-space areas/normals for texel density and orientation
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    uv = bm.loops.layers.uv[uv_name]

    islands = bmesh_utils.bmesh_linked_uv_islands(bm, uv)

    def uv_area(faces):
        total = 0.0
        for f in faces:
            pts = [l[uv].uv for l in f.loops]
            for i in range(1, len(pts) - 1):
                total += abs((pts[i].x - pts[0].x) * (pts[i + 1].y - pts[0].y) - (pts[i + 1].x - pts[0].x) * (pts[i].y - pts[0].y)) / 2
        return total

    islands = sorted(islands, key=uv_area, reverse=True)  # id 1 = largest island
    face_island = {}
    for k, faces in enumerate(islands, start=1):
        for f in faces:
            face_island[f.index] = k

    def px(u, v):
        return (u * W, (1.0 - v) * H)

    tris, tri_ids, tri_normals, tri_up = [], [], [], []
    flipped = 0
    out_of_bounds = 0
    for f in bm.faces:
        pts = [l[uv].uv.copy() for l in f.loops]
        if any(p.x < -1e-4 or p.x > 1 + 1e-4 or p.y < -1e-4 or p.y > 1 + 1e-4 for p in pts):
            out_of_bounds += 1
        signed = 0.0
        for i in range(1, len(pts) - 1):
            signed += (pts[i].x - pts[0].x) * (pts[i + 1].y - pts[0].y) - (pts[i + 1].x - pts[0].x) * (pts[i].y - pts[0].y)
        if signed < 0:
            flipped += 1
        up_deg = face_up_angle(f, uv)
        for i in range(1, len(pts) - 1):
            tris.append((px(*pts[0]), px(*pts[i]), px(*pts[i + 1])))
            tri_ids.append(face_island[f.index])
            tri_normals.append(tuple(f.normal))
            tri_up.append(up_deg)

    ids, overlap, normals, up_angle = raster_triangles(tris, tri_ids, tri_normals, tri_up, W, H)

    info_islands = []
    for k, faces in enumerate(islands, start=1):
        us = [l[uv].uv.x for f in faces for l in f.loops]
        vs = [l[uv].uv.y for f in faces for l in f.loops]
        area3d = sum(f.calc_area() for f in faces)
        normal = sum((f.normal * f.calc_area() for f in faces), start=faces[0].normal * 0)
        normal = normal.normalized() if normal.length > 1e-9 else normal
        area_px = uv_area(faces) * W * H
        info_islands.append({
            "id": k,
            "faces": len(faces),
            "bbox": [
                max(0, math.floor(min(us) * W)), max(0, math.floor((1 - max(vs)) * H)),
                min(W, math.ceil(max(us) * W)), min(H, math.ceil((1 - min(vs)) * H)),
            ],
            "area_px": round(area_px, 1),
            "texels": int((ids == k).sum()),
            "area_3d": round(area3d, 6),
            "texel_density": round(math.sqrt(area_px / area3d), 3) if area3d > 1e-12 else None,
            "normal": [round(c, 3) for c in normal],  # area-weighted world normal: which way the island faces
            "up_rotation": None,  # filled below from the up-angle map
        })

    # UV seams: mesh edges shared by two faces whose UVs don't match on both sides.
    seams = []
    for e in bm.edges:
        loops = list(e.link_loops)
        if len(loops) != 2:
            continue
        side = []
        for l in loops:
            a, b = l, l.link_loop_next
            side.append({a.vert.index: a[uv].uv.copy(), b.vert.index: b[uv].uv.copy()})
        v0, v1 = e.verts[0].index, e.verts[1].index
        if (side[0][v0] - side[1][v0]).length < 1e-5 and (side[0][v1] - side[1][v1]).length < 1e-5:
            continue
        seams.append({
            "edge": e.index,
            "islands": [face_island[loops[0].face.index], face_island[loops[1].face.index]],
            "a": [list(px(*side[0][v0])), list(px(*side[0][v1]))],
            "b": [list(px(*side[1][v0])), list(px(*side[1][v1]))],
            "length_3d": round(e.calc_length(), 6),
        })

    for isl in info_islands:
        a = up_angle[ids == isl["id"]].astype(np.float32)
        a = a[~np.isnan(a)]
        if len(a):
            q = (np.round(a / 90) % 4).astype(int) * 90
            vals, counts = np.unique(q, return_counts=True)
            isl["up_rotation"] = {int(v): round(float(c) / len(q), 3) for v, c in zip(vals, counts)}

    bm.free()
    eval_obj.to_mesh_clear()

    os.makedirs(opts["out_dir"], exist_ok=True)
    np.savez_compressed(os.path.join(opts["out_dir"], "uv_data.npz"), islands=ids, overlap=overlap, normals=normals, up_angle=up_angle)
    dens = [i["texel_density"] for i in info_islands if i["texel_density"]]
    info = {
        "blend_file": bpy.data.filepath,
        "object": obj.name,
        "mesh": obj.data.name,
        "uv_map": uv_name,
        "uv_maps": uv_names,
        "size": [W, H],
        "textures": textures,
        "faces": len(face_island),
        "islands": info_islands,
        "seams": seams,
        "sanity": {
            "island_count": len(islands),
            "coverage": round(float((ids > 0).mean()), 4),
            "overlap_texels": int(overlap.sum()),
            "faces_outside_0_1": out_of_bounds,
            "flipped_faces": flipped,
            "texel_density_cv": round(float(np.std(dens) / np.mean(dens)), 3) if dens else None,
            "udim": any(t["source"] == "TILED" for t in textures),
        },
    }
    with open(os.path.join(opts["out_dir"], "uv_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"UV_EXPORT_OK {json.dumps({'out_dir': opts['out_dir'], 'islands': len(islands), 'size': [W, H]})}")


main()
