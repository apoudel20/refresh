"""
Connector to a Blender MCP socket server.

Two wire protocols are supported (``MCPConfig.protocol``):

* ``"lab"`` (default) - the official Blender Lab MCP extension (Blender 5.1+,
  ``extensions/lab_blender_org/mcp``). Requests are
  ``{"type": "execute", "code": "...", "strict_json": false}`` terminated by a NUL
  byte; replies are ``{"status": "ok", "result": {...}}`` or
  ``{"status": "error", "message": ...}``, also NUL-terminated. Executed code
  returns data by assigning a dict to ``result``.
* ``"community"`` - the community blender-mcp add-on: raw JSON
  ``{"type": "<command>", "params": {...}}`` with ``{"status": "success"}`` replies.

Every high-level method builds bpy code and runs it through ``_exec``; code sets
``__result__`` and the connector maps it onto whichever protocol is in use.

Harness helpers (``setup_stage``, ``snapshot``, ``restore``, ``render_stage``,
``export_glb``) keep a locked camera/light rig in a ``RefreshStage`` collection and
move the model between nodes as .blend datablock files, so the open Blender file is
never replaced (no ``open_mainfile`` from inside the server's timer).
"""

from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import dataclass, field
from typing import Any

STAGE_COLLECTION = "RefreshStage"


@dataclass
class MCPConfig:
    mode: str = "socket"  # kept for API compat
    host: str = field(default_factory=lambda: os.getenv("BLENDER_MCP_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("BLENDER_MCP_PORT", "9876")))
    timeout: float = 600.0  # renders can take minutes
    protocol: str = field(default_factory=lambda: os.getenv("BLENDER_MCP_PROTOCOL", "lab"))  # "lab" | "community"


class BlenderMCPConnector:
    def __init__(self, config: MCPConfig | None = None):
        self.config = config or MCPConfig()
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self.last_stdout = ""

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.config.timeout)
        self._sock.connect((self.config.host, self.config.port))

    def disconnect(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __enter__(self):
        return self  # connections are opened per request

    def __exit__(self, *_):
        self.disconnect()

    def ping(self) -> bool:
        try:
            return self._exec("__result__ = 'pong'") == "pong"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Low-level send/receive
    # ------------------------------------------------------------------

    def _roundtrip(self, payload: bytes, terminator: bytes | None) -> dict[str, Any]:
        """One request on a fresh connection. terminator=None: read until the JSON parses."""
        with self._lock:
            last_err: Exception | None = None
            for _attempt in range(2):
                try:
                    self.connect()
                    self._sock.sendall(payload)
                    buf = b""
                    while True:
                        chunk = self._sock.recv(65536)
                        if not chunk:
                            if terminator is None and buf:
                                return json.loads(buf.decode("utf-8"))
                            raise ConnectionError("Blender MCP socket closed before a full reply")
                        buf += chunk
                        if terminator is not None:
                            if terminator in buf:
                                return json.loads(buf[: buf.index(terminator)].decode("utf-8"))
                        else:
                            try:
                                return json.loads(buf.decode("utf-8"))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
                except OSError as exc:
                    last_err = exc
                finally:
                    self.disconnect()
            raise ConnectionError(
                f"Blender MCP server not reachable at {self.config.host}:{self.config.port} ({last_err}). "
                "In Blender: enable the MCP extension and start its server."
            )

    def _exec(self, code: str) -> Any:
        """Run bpy code inside Blender; returns the value the code assigned to ``__result__``."""
        if self.config.protocol == "community":
            resp = self._roundtrip(json.dumps({"type": "execute_code", "params": {"code": code}}).encode(), None)
            if resp.get("status") == "error":
                raise RuntimeError(f"Blender error: {resp.get('message')}")
            return resp.get("result")

        wrapped = (
            code
            + "\n\ntry:\n    result = {'value': __result__}\nexcept NameError:\n"
            "    result = {'value': result if result else None}\n"
        )
        req = {"type": "execute", "code": wrapped, "strict_json": False}
        resp = self._roundtrip((json.dumps(req) + "\0").encode("utf-8"), b"\0")
        self.last_stdout = resp.get("stdout", "") or ""
        if resp.get("status") != "ok":
            detail = resp.get("message", "")
            if resp.get("stderr"):
                detail += f"\nstderr: {resp['stderr'][-2000:]}"
            raise RuntimeError(f"Blender error: {detail}")
        res = resp.get("result") or {}
        return res.get("value") if isinstance(res, dict) else res

    # ------------------------------------------------------------------
    # Generic tools
    # ------------------------------------------------------------------

    def execute_python(self, code: str) -> Any:
        """Arbitrary bpy code (LLM-authored). Returns __result__/result plus captured stdout."""
        value = self._exec(code)
        return {"result": value, "stdout": self.last_stdout[-4000:]} if self.last_stdout else value

    def get_scene_info(self) -> dict[str, Any]:
        return self._exec(f"""
import bpy
stage = bpy.data.collections.get({STAGE_COLLECTION!r})
stage_objs = set(stage.all_objects) if stage else set()
__result__ = {{
    "scene": bpy.context.scene.name,
    "objects": [{{"name": o.name, "type": o.type, "location": [round(v, 4) for v in o.location],
                 "rotation": [round(v, 4) for v in o.rotation_euler], "scale": [round(v, 4) for v in o.scale],
                 "dimensions": [round(v, 4) for v in o.dimensions], "stage": o in stage_objs,
                 "materials": [s.material.name for s in getattr(o, "material_slots", []) if s.material]}}
                for o in bpy.context.scene.objects],
    "render_engine": bpy.context.scene.render.engine,
}}
""")

    def get_object_info(self, name: str) -> dict[str, Any]:
        return self._exec(f"""
import bpy
o = bpy.data.objects.get({name!r})
__result__ = None if o is None else {{"name": o.name, "type": o.type, "location": list(o.location),
    "dimensions": list(o.dimensions), "modifiers": [m.name for m in o.modifiers],
    "vertices": len(o.data.vertices) if o.type == 'MESH' else None}}
""")

    def import_file(self, path: str, file_format: str | None = None) -> dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} does not exist (earlier agents' work is already in the scene; "
                                    "there are no model files to import)")
        fmt = (file_format or path.rsplit(".", 1)[-1]).lower()
        ops = {
            "ply": "bpy.ops.wm.ply_import(filepath=r'{p}')",
            "obj": "bpy.ops.wm.obj_import(filepath=r'{p}')",
            "fbx": "bpy.ops.import_scene.fbx(filepath=r'{p}')",
            "gltf": "bpy.ops.import_scene.gltf(filepath=r'{p}')",
            "glb": "bpy.ops.import_scene.gltf(filepath=r'{p}')",
            "stl": "bpy.ops.wm.stl_import(filepath=r'{p}')",
        }
        op = ops.get(fmt, "bpy.ops.wm.ply_import(filepath=r'{p}')")
        return self._exec(
            f"import bpy\nbefore = set(bpy.data.objects)\n{op.format(p=path)}\n"
            "__result__ = {'imported': [o.name for o in bpy.data.objects if o not in before]}"
        )

    def export_file(self, path: str, file_format: str | None = None, object_names: list[str] | None = None) -> dict[str, Any]:
        fmt = (file_format or path.rsplit(".", 1)[-1]).lower()
        select_code = ""
        if object_names:
            select_code = (
                "bpy.ops.object.select_all(action='DESELECT')\n"
                f"[bpy.data.objects[n].select_set(True) for n in {object_names!r} if n in bpy.data.objects]\n"
            )
        sel = bool(object_names)
        ops = {
            "ply": f"bpy.ops.wm.ply_export(filepath=r'{path}')",
            "obj": f"bpy.ops.wm.obj_export(filepath=r'{path}')",
            "fbx": f"bpy.ops.export_scene.fbx(filepath=r'{path}', use_selection={sel})",
            "gltf": f"bpy.ops.export_scene.gltf(filepath=r'{path}', use_selection={sel})",
            "glb": f"bpy.ops.export_scene.gltf(filepath=r'{path}', export_format='GLB', use_selection={sel})",
            "stl": f"bpy.ops.wm.stl_export(filepath=r'{path}')",
        }
        op = ops.get(fmt, ops["obj"])
        return self._exec(f"import bpy, os\nos.makedirs(os.path.dirname(r'{path}') or '.', exist_ok=True)\n{select_code}{op}\n__result__ = r'{path}'")

    def set_material(self, object_name: str, texture_path: str, mapping: str = "UV") -> dict[str, Any]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj is None:
    raise ValueError(f"Object {object_name!r} not found")
mat = bpy.data.materials.new(name="GenMat_" + obj.name)
mat.use_nodes = True
tree = mat.node_tree
tree.nodes.clear()
tex_node = tree.nodes.new('ShaderNodeTexImage')
bsdf_node = tree.nodes.new('ShaderNodeBsdfPrincipled')
out_node = tree.nodes.new('ShaderNodeOutputMaterial')
tex_node.image = bpy.data.images.load({texture_path!r}, check_existing=True)
if {mapping!r} != 'UV':
    coord = tree.nodes.new('ShaderNodeTexCoord')
    tree.links.new(coord.outputs['Generated' if {mapping!r} == 'GENERATED' else 'Object'], tex_node.inputs['Vector'])
tree.links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
tree.links.new(bsdf_node.outputs['BSDF'], out_node.inputs['Surface'])
if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)
__result__ = mat.name
""")

    def render(self, output_path: str, camera_angles: list[tuple[float, float, float]] | None = None,
               resolution: tuple[int, int] = (512, 512), engine: str = "EEVEE") -> list[str]:
        """Orbit renders for the agent's own inspection. Uses temporary 'RenderCam' cameras and restores the
        stage camera afterwards; scored renders always come from ``render_stage``."""
        angles = camera_angles or [(0.0, 20.0, 0.0)]
        paths: list[str] = []
        for i, ang in enumerate(angles):
            azimuth, elevation = (list(ang) + [0.0, 20.0])[:2]
            out = f"{output_path}/render_angle{i}.png"
            self._exec(f"""
import bpy, math, mathutils, os
os.makedirs(os.path.dirname({out!r}), exist_ok=True)
scene = bpy.context.scene
prev_cam = scene.camera
scene.render.resolution_x, scene.render.resolution_y = {resolution[0]}, {resolution[1]}
{_ENGINE_SNIPPET.format(engine=engine)}
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = {out!r}
objs = [o for o in scene.objects if o.type == 'MESH' and not any(c.name == {STAGE_COLLECTION!r} for c in o.users_collection)]
if objs:
    pts = [o.matrix_world @ mathutils.Vector(c) for o in objs for c in o.bound_box]
    lo = mathutils.Vector([min(p[k] for p in pts) for k in range(3)]); hi = mathutils.Vector([max(p[k] for p in pts) for k in range(3)])
    center, radius = (lo + hi) / 2, max((hi - lo).length, 0.5) * 1.4
else:
    center, radius = mathutils.Vector((0, 0, 0.5)), 4.0
az, el = math.radians({azimuth}), math.radians({elevation})
eye = center + mathutils.Vector((radius * math.sin(az) * math.cos(el), -radius * math.cos(az) * math.cos(el), radius * math.sin(el)))
for o in [o for o in bpy.data.objects if o.name.startswith("RenderCam")]:
    bpy.data.objects.remove(o, do_unlink=True)
cam = bpy.data.cameras.new("RenderCam")
cam_obj = bpy.data.objects.new("RenderCam", cam)
scene.collection.objects.link(cam_obj)
cam_obj.location = eye
cam_obj.rotation_euler = (center - eye).to_track_quat('-Z', 'Y').to_euler()
scene.camera = cam_obj
bpy.ops.render.render(write_still=True)
scene.camera = prev_cam
bpy.data.objects.remove(cam_obj, do_unlink=True)
__result__ = {out!r}
""")
            paths.append(out)
        return paths

    def get_depth_map(self, output_path: str, resolution: tuple[int, int] = (512, 512)) -> str:
        """Depth pass from the stage camera, normalised to 0..1 and saved as PNG."""
        return self._exec(f"""
import bpy, os
import numpy as np
scene = bpy.context.scene
os.makedirs(os.path.dirname({output_path!r}) or '.', exist_ok=True)
scene.render.resolution_x, scene.render.resolution_y = {resolution[0]}, {resolution[1]}
scene.view_layers[0].use_pass_z = True
scene.render.use_compositing = True
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()
rl = tree.nodes.new('CompositorNodeRLayers')
norm = tree.nodes.new('CompositorNodeNormalize')
viewer = tree.nodes.new('CompositorNodeViewer')
tree.links.new(rl.outputs['Depth'], norm.inputs[0])
tree.links.new(norm.outputs[0], viewer.inputs[0])
bpy.ops.render.render()
img = bpy.data.images['Viewer Node']
px = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], 4)
out = bpy.data.images.new("depth_tmp", img.size[0], img.size[1])
out.pixels = px.ravel()
out.filepath_raw = {output_path!r}
out.file_format = 'PNG'
out.save()
bpy.data.images.remove(out)
tree.nodes.clear()
scene.use_nodes = False
__result__ = {output_path!r}
""")

    def get_vertex_positions(self, object_name: str) -> list[list[float]]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj and obj.type == 'MESH':
    __result__ = [[round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4)] for v in obj.data.vertices[:2000]]
else:
    __result__ = []
""") or []

    def get_topology_stats(self, object_name: str) -> dict[str, Any]:
        return self._exec(f"""
import bpy, bmesh
obj = bpy.data.objects.get({object_name!r})
if obj and obj.type == 'MESH':
    m = obj.data
    bm = bmesh.new(); bm.from_mesh(m)
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    quads = sum(1 for f in bm.faces if len(f.verts) == 4)
    bm.free()
    __result__ = {{"vertices": len(m.vertices), "edges": len(m.edges), "faces": len(m.polygons),
                   "quads": quads, "non_manifold_edges": non_manifold}}
else:
    __result__ = {{}}
""") or {}

    def apply_subdivision(self, object_name: str, levels: int = 2) -> dict[str, Any]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj:
    mod = obj.modifiers.new("Subd", "SUBSURF")
    mod.levels = {levels}
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.modifier_apply(modifier=mod.name)
__result__ = {{"object": {object_name!r}, "applied": obj is not None}}
""")

    def smooth_mesh(self, object_name: str, iterations: int = 5, factor: float = 0.5) -> dict[str, Any]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj:
    mod = obj.modifiers.new("Smooth", "SMOOTH")
    mod.iterations = {iterations}
    mod.factor = {factor}
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.modifier_apply(modifier=mod.name)
__result__ = {{"object": {object_name!r}, "applied": obj is not None}}
""")

    def unwrap_uv(self, object_name: str, method: str = "SMART_PROJECT") -> dict[str, Any]:
        return self._exec(f"""
import bpy, bmesh
obj = bpy.data.objects.get({object_name!r})
if obj and obj.type == 'MESH':
    bpy.context.view_layer.objects.active = obj
    with bpy.context.temp_override(object=obj, active_object=obj, edit_object=obj):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        if {method!r} == 'SMART_PROJECT':
            bpy.ops.uv.smart_project()
        else:
            bpy.ops.uv.unwrap(method={method!r})
        bpy.ops.object.mode_set(mode='OBJECT')
__result__ = {{"object": {object_name!r}, "uv_layers": [l.name for l in obj.data.uv_layers] if obj else []}}
""")

    def delete_object(self, object_name: str) -> dict[str, Any]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj:
    bpy.data.objects.remove(obj, do_unlink=True)
__result__ = obj is not None
""")

    def new_scene(self) -> dict[str, Any]:
        """Delete every object except the stage rig, leaving a clean slate (never a factory reset)."""
        return self._exec(_CLEAR_MODEL + "\n__result__ = {'cleared': True}")

    def clear_scene(self) -> dict[str, Any]:
        return self.new_scene()

    def save_blend(self, path: str) -> dict[str, Any]:
        """Save a copy of the current .blend (the open file keeps its own path)."""
        return self._exec(f"""
import bpy, os
os.makedirs(os.path.dirname({path!r}), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath={path!r}, copy=True)
__result__ = {path!r}
""")

    # ------------------------------------------------------------------
    # Harness helpers: stage rig, node hand-off, scored renders
    # ------------------------------------------------------------------

    def setup_stage(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Create or reset the locked camera + lights in the RefreshStage collection.

        spec: {"resolution": [w, h], "camera": {"location": [x,y,z], "target": [x,y,z], "lens": mm},
               "lights": [{"type": "SUN"|"AREA", "location": [...], "energy": float}], "engine": "EEVEE"}
        """
        return self._exec(f"""
import bpy, mathutils
spec = {json.dumps(spec)}
scene = bpy.context.scene
col = bpy.data.collections.get({STAGE_COLLECTION!r})
if col is None:
    col = bpy.data.collections.new({STAGE_COLLECTION!r})
    scene.collection.children.link(col)
for o in list(col.objects):
    bpy.data.objects.remove(o, do_unlink=True)
cam_spec = spec.get("camera", {{}})
cam = bpy.data.cameras.new("StageCam")
cam.lens = cam_spec.get("lens", 50.0)
cam_obj = bpy.data.objects.new("StageCam", cam)
col.objects.link(cam_obj)
loc = mathutils.Vector(cam_spec.get("location", [0.0, -6.0, 1.2]))
target = mathutils.Vector(cam_spec.get("target", [0.0, 0.0, 1.0]))
cam_obj.location = loc
cam_obj.rotation_euler = (target - loc).to_track_quat('-Z', 'Y').to_euler()
cam_obj.hide_select = True
scene.camera = cam_obj
for i, L in enumerate(spec.get("lights", [])):
    ld = bpy.data.lights.new(f"StageLight{{i}}", L.get("type", "AREA"))
    ld.energy = L.get("energy", 500.0)
    if ld.type == 'AREA':
        ld.size = L.get("size", 4.0)
    lo = bpy.data.objects.new(f"StageLight{{i}}", ld)
    col.objects.link(lo)
    lo.location = L.get("location", [4.0, -5.0, 5.0])
    lo.rotation_euler = (target - lo.location).to_track_quat('-Z', 'Y').to_euler()
    lo.hide_select = True
w, h = spec.get("resolution", [768, 768])
scene.render.resolution_x, scene.render.resolution_y = int(w), int(h)
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
{_ENGINE_SNIPPET.format(engine=spec.get("engine", "EEVEE"))}
if scene.world is None:
    scene.world = bpy.data.worlds.new("StageWorld")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[1].default_value = spec.get("world_strength", 0.6)
__result__ = {{"camera": cam_obj.name, "lights": len(spec.get("lights", [])), "resolution": [w, h]}}
""")

    def snapshot(self, path: str) -> dict[str, Any]:
        """Write every non-stage object (with its data, materials, images) to a .blend file."""
        return self._exec(f"""
import bpy, os
os.makedirs(os.path.dirname({path!r}), exist_ok=True)
stage = bpy.data.collections.get({STAGE_COLLECTION!r})
stage_objs = set(stage.all_objects) if stage else set()
objs = {{o for o in bpy.context.scene.objects if o not in stage_objs and not o.name.startswith("RenderCam")}}
bpy.data.libraries.write({path!r}, objs, fake_user=True)
__result__ = {{"path": {path!r}, "objects": sorted(o.name for o in objs)}}
""")

    def restore(self, path: str | None, clear: bool = True) -> dict[str, Any]:
        """Clear the model (unless clear=False, used to merge several parents) and append a snapshot's objects."""
        load = ""
        if path:
            load = f"""
with bpy.data.libraries.load({path!r}, link=False) as (src, dst):
    dst.objects = list(src.objects)
for o in dst.objects:
    if o is not None:
        bpy.context.scene.collection.objects.link(o)
        o.use_fake_user = False
"""
        return self._exec(
            "import bpy\n" + (_CLEAR_MODEL if clear else "") + load
            + "\n__result__ = {'objects': [o.name for o in bpy.context.scene.objects]}"
        )

    def render_stage(self, output_path: str) -> str:
        """Render from the locked stage camera (re-selected every time, so agents can't move the scoring view)."""
        return self._exec(f"""
import bpy, os
scene = bpy.context.scene
cam = bpy.data.objects.get("StageCam")
if cam is None:
    raise RuntimeError("stage not set up: call setup_stage first")
scene.camera = cam
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
os.makedirs(os.path.dirname({output_path!r}) or '.', exist_ok=True)
scene.render.filepath = {output_path!r}
bpy.ops.render.render(write_still=True)
__result__ = {output_path!r}
""")

    def scene_images(self) -> list[dict[str, Any]]:
        """Image datablocks and the files they load (the reference-photo guard checks these)."""
        return self._exec(
            "import bpy\n"
            "__result__ = [{'name': i.name, 'path': bpy.path.abspath(i.filepath) if i.filepath else '', "
            "'users': i.users} for i in bpy.data.images if i.source in ('FILE', 'SEQUENCE', 'MOVIE')]"
        ) or []

    def turntable(self, out_dir: str, views: int = 8, size: int = 256, elevation: float = 15.0) -> dict[str, Any]:
        """Orthographic views all around the model (azimuth 0 = the stage camera's side), rendered flat:
        front faces slate blue, back faces red. A closed solid shows no red from any side; an open shell or a
        relief shows red from behind and a thin sliver from the side. Returns the PNG paths."""
        code = _TURNTABLE.replace("__OUT__", repr(out_dir)).replace("__VIEWS__", str(int(views)))
        code = code.replace("__SIZE__", str(int(size))).replace("__ELEV__", repr(float(elevation)))
        code = code.replace("__STAGE__", repr(STAGE_COLLECTION))
        return self._exec(code)

    def render_preview(self, output_path: str, percent: int = 50, samples: int = 8) -> str:
        """Quick, low-res render of the stage view (for live progress); every setting is restored."""
        return self._exec(f"""
import bpy, os
scene = bpy.context.scene
cam = bpy.data.objects.get("StageCam")
if cam is None:
    raise RuntimeError("stage not set up")
r = scene.render
saved = (scene.camera, r.resolution_percentage, r.film_transparent, r.filepath,
         r.image_settings.file_format, r.image_settings.color_mode, getattr(scene.eevee, 'taa_render_samples', None))
try:
    scene.camera = cam
    r.resolution_percentage = {int(percent)}
    r.film_transparent = True
    r.image_settings.file_format = 'PNG'
    r.image_settings.color_mode = 'RGBA'
    if saved[6] is not None:
        scene.eevee.taa_render_samples = {int(samples)}
    os.makedirs(os.path.dirname({output_path!r}) or '.', exist_ok=True)
    r.filepath = {output_path!r}
    bpy.ops.render.render(write_still=True)
finally:
    (scene.camera, r.resolution_percentage, r.film_transparent, r.filepath,
     r.image_settings.file_format, r.image_settings.color_mode) = saved[:6]
    if saved[6] is not None:
        scene.eevee.taa_render_samples = saved[6]
__result__ = {output_path!r}
""")

    def export_glb(self, path: str) -> str:
        """Export the model (everything outside the stage rig) as a binary glTF for the web viewer."""
        return self._exec(f"""
import bpy, os
os.makedirs(os.path.dirname({path!r}), exist_ok=True)
stage = bpy.data.collections.get({STAGE_COLLECTION!r})
stage_objs = set(stage.all_objects) if stage else set()
for o in bpy.context.scene.objects:
    o.select_set(o not in stage_objs and o.type in ('MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'EMPTY'))
bpy.ops.export_scene.gltf(filepath={path!r}, export_format='GLB', use_selection=True, export_apply=True)
__result__ = {path!r}
""")


_TURNTABLE = """
import bpy, math, os
from mathutils import Vector
scene = bpy.context.scene
os.makedirs(__OUT__, exist_ok=True)
stage = bpy.data.collections.get(__STAGE__)
stage_objs = set(stage.all_objects) if stage else set()
objs = [o for o in scene.objects if o.type in ('MESH', 'CURVE', 'SURFACE', 'META', 'FONT')
        and o not in stage_objs and not o.hide_render and o.visible_get()]
if not objs:
    __result__ = {'views': [], 'empty': True}
else:
    dg = bpy.context.evaluated_depsgraph_get()
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.evaluated_get(dg).bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-3)
    base = (scene.camera.location - center) if scene.camera else Vector((0.0, -1.0, 0.0))
    az0 = math.atan2(base.y, base.x)
    mat = bpy.data.materials.get('RefreshSolidity') or bpy.data.materials.new('RefreshSolidity')
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    mix = nt.nodes.new('ShaderNodeMixShader')
    front = nt.nodes.new('ShaderNodeEmission')
    back = nt.nodes.new('ShaderNodeEmission')
    facing = nt.nodes.new('ShaderNodeLayerWeight')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (0.62, 0.72, 0.86, 1.0)
    ramp.color_ramp.elements[1].color = (0.16, 0.22, 0.32, 1.0)
    nt.links.new(facing.outputs['Facing'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], front.inputs['Color'])
    back.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1.0)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(geo.outputs['Backfacing'], mix.inputs['Fac'])
    nt.links.new(front.outputs['Emission'], mix.inputs[1])
    nt.links.new(back.outputs['Emission'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    mat.use_backface_culling = False
    cam_data = bpy.data.cameras.new('RefreshTurntableCam')
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = radius * 2.2
    cam_data.clip_start = 0.001
    cam_data.clip_end = radius * 8 + 10
    cam = bpy.data.objects.new('RefreshTurntableCam', cam_data)
    scene.collection.objects.link(cam)
    vl = bpy.context.view_layer
    r, im = scene.render, scene.render.image_settings
    saved = (scene.camera, r.engine, r.resolution_x, r.resolution_y, r.resolution_percentage, r.film_transparent,
             r.filepath, scene.view_settings.view_transform, scene.view_settings.look, vl.material_override,
             im.file_format, im.color_mode)
    samples = getattr(scene.eevee, 'taa_render_samples', None)
    paths = []
    try:
        for name in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
            try:
                r.engine = name
                break
            except TypeError:
                continue
        try:
            scene.eevee.taa_render_samples = 1
        except AttributeError:
            pass
        r.resolution_x = r.resolution_y = __SIZE__
        r.resolution_percentage = 100
        r.film_transparent = True
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        vl.material_override = mat
        im.file_format = 'PNG'
        im.color_mode = 'RGBA'
        scene.camera = cam
        el = math.radians(__ELEV__)
        for i in range(__VIEWS__):
            az = az0 + 2 * math.pi * i / __VIEWS__
            d = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
            cam.location = center + d * (radius * 4)
            cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
            path = os.path.join(__OUT__, 'view_%02d.png' % i)
            r.filepath = path
            bpy.ops.render.render(write_still=True)
            paths.append({'path': path, 'azimuth': round(360.0 * i / __VIEWS__, 1)})
    finally:
        (scene.camera, r.engine, r.resolution_x, r.resolution_y, r.resolution_percentage, r.film_transparent,
         r.filepath, scene.view_settings.view_transform, scene.view_settings.look, vl.material_override,
         im.file_format, im.color_mode) = saved
        bpy.data.objects.remove(cam, do_unlink=True)
        bpy.data.cameras.remove(cam_data)
        if samples is not None:
            scene.eevee.taa_render_samples = samples
    __result__ = {'views': paths, 'empty': False, 'size': [round(v, 4) for v in (hi - lo)]}
"""

_ENGINE_SNIPPET = """_engine = {engine!r}
if _engine in ('EEVEE', 'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
    for _name in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        try:
            scene.render.engine = _name
            break
        except TypeError:
            continue
    try:
        scene.eevee.taa_render_samples = 16
    except AttributeError:
        pass
elif _engine == 'CYCLES':
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
else:
    scene.render.engine = _engine"""

_CLEAR_MODEL = f"""
import bpy
stage = bpy.data.collections.get({STAGE_COLLECTION!r})
stage_objs = set(stage.all_objects) if stage else set()
for o in [o for o in bpy.context.scene.objects if o not in stage_objs]:
    bpy.data.objects.remove(o, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.curves):
    for block in [b for b in coll if b.users == 0]:
        coll.remove(block)
"""
