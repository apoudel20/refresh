"""
Connector to the mcp-for-blender addon socket server.

Protocol (per addon source):
  Send:    {"type": "<command>", "params": {...}}  — raw JSON, no framing
  Receive: {"status": "success", "result": ...}
        or {"status": "error",   "message": ...}

Native commands: get_scene_info, get_object_info, execute_code
Everything else is implemented via execute_code + bpy Python.
"""

import json
import socket
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class MCPConfig:
    mode: str = "socket"        # only "socket" is used; kept for API compat
    host: str = "localhost"
    port: int = 9876
    timeout: float = 30.0


class BlenderMCPConnector:
    def __init__(self, config: MCPConfig | None = None):
        self.config = config or MCPConfig()
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

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
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------
    # Low-level send/receive
    # ------------------------------------------------------------------

    def _send(self, cmd_type: str, params: dict[str, Any]) -> Any:
        payload = json.dumps({"type": cmd_type, "params": params}).encode("utf-8")

        with self._lock:
            # Reconnect if the socket dropped (addon closes connection per-call)
            for attempt in range(2):
                try:
                    if self._sock is None:
                        self.connect()
                    self._sock.sendall(payload)
                    buf = b""
                    while True:
                        chunk = self._sock.recv(65536)
                        if not chunk:
                            raise ConnectionError("Blender MCP socket closed")
                        buf += chunk
                        try:
                            response = json.loads(buf.decode("utf-8"))
                            break
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                    break  # success
                except OSError:
                    # Socket died — close and retry once with a fresh connection
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                    if attempt == 1:
                        raise

        if response.get("status") == "error":
            raise RuntimeError(f"Blender error: {response.get('message')}")
        return response.get("result")

    def _exec(self, code: str) -> Any:
        """Run Python inside Blender and return the result of __result__."""
        return self._send("execute_code", {"code": code})

    # ------------------------------------------------------------------
    # High-level tool methods
    # ------------------------------------------------------------------

    def execute_python(self, code: str) -> Any:
        return self._send("execute_code", {"code": code})

    def get_scene_info(self) -> dict[str, Any]:
        return self._send("get_scene_info", {})

    def get_object_info(self, name: str) -> dict[str, Any]:
        return self._send("get_object_info", {"name": name})

    def import_file(self, path: str, file_format: str | None = None) -> dict[str, Any]:
        fmt = (file_format or path.rsplit(".", 1)[-1]).lower()
        ops = {
            "ply":  "bpy.ops.wm.ply_import(filepath=r'{p}')",
            "obj":  "bpy.ops.wm.obj_import(filepath=r'{p}')",
            "fbx":  "bpy.ops.import_scene.fbx(filepath=r'{p}')",
            "gltf": "bpy.ops.import_scene.gltf(filepath=r'{p}')",
            "glb":  "bpy.ops.import_scene.gltf(filepath=r'{p}')",
            "stl":  "bpy.ops.wm.stl_import(filepath=r'{p}')",
        }
        op = ops.get(fmt, "bpy.ops.wm.ply_import(filepath=r'{p}')")
        code = f"import bpy; {op.format(p=path)}"
        return self._exec(code)

    def export_file(
        self,
        path: str,
        file_format: str | None = None,
        object_names: list[str] | None = None,
    ) -> dict[str, Any]:
        fmt = (file_format or path.rsplit(".", 1)[-1]).lower()
        select_code = ""
        if object_names:
            names_repr = repr(object_names)
            select_code = (
                "import bpy;"
                "bpy.ops.object.select_all(action='DESELECT');"
                f"[bpy.data.objects[n].select_set(True) for n in {names_repr} if n in bpy.data.objects];"
            )
        ops = {
            "ply":  f"bpy.ops.wm.ply_export(filepath=r'{path}')",
            "obj":  f"bpy.ops.wm.obj_export(filepath=r'{path}')",
            "fbx":  f"bpy.ops.export_scene.fbx(filepath=r'{path}', use_selection={bool(object_names)})",
            "gltf": f"bpy.ops.export_scene.gltf(filepath=r'{path}', use_selection={bool(object_names)})",
            "glb":  f"bpy.ops.export_scene.gltf(filepath=r'{path}', use_selection={bool(object_names)})",
            "stl":  f"bpy.ops.wm.stl_export(filepath=r'{path}')",
        }
        op = ops.get(fmt, ops["obj"])
        return self._exec(f"import bpy; {select_code}{op}")

    def set_material(self, object_name: str, texture_path: str, mapping: str = "UV") -> dict[str, Any]:
        code = f"""
import bpy, os
obj = bpy.data.objects.get({object_name!r})
if obj is None:
    raise ValueError(f"Object {object_name!r} not found")
mat = bpy.data.materials.new(name="GenMat_" + obj.name)
mat.use_nodes = True
tree = mat.node_tree
tree.nodes.clear()
tex_node  = tree.nodes.new('ShaderNodeTexImage')
bsdf_node = tree.nodes.new('ShaderNodeBsdfPrincipled')
out_node  = tree.nodes.new('ShaderNodeOutputMaterial')
img = bpy.data.images.load({texture_path!r})
tex_node.image = img
tree.links.new(tex_node.outputs['Color'],  bsdf_node.inputs['Base Color'])
tree.links.new(bsdf_node.outputs['BSDF'],  out_node.inputs['Surface'])
if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)
"""
        return self._exec(code)

    def render(
        self,
        output_path: str,
        camera_angles: list[tuple[float, float, float]] | None = None,
        resolution: tuple[int, int] = (512, 512),
        engine: str = "EEVEE",
    ) -> list[str]:
        # camera_angles is [[azimuth, elevation, roll], ...] in degrees.
        # azimuth orbits around the object (0=front, 90=right, 180=back, 270=left).
        # elevation tilts up from horizontal (20 gives a slight top-down view).
        angles = camera_angles or [(0.0, 20.0, 0.0)]
        paths: list[str] = []
        for i, (azimuth, elevation, roll) in enumerate(angles):
            out = f"{output_path}/render_angle{i}.png"
            code = f"""
import bpy, math, mathutils, os
os.makedirs(os.path.dirname({out!r}), exist_ok=True)
scene = bpy.context.scene
scene.render.resolution_x = {resolution[0]}
scene.render.resolution_y = {resolution[1]}
_engine = {engine!r}
if _engine in ('EEVEE', 'BLENDER_EEVEE'):
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except Exception:
        scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 1
    scene.eevee.use_bloom = False
    scene.eevee.use_ssr = False
    scene.eevee.use_motion_blur = False
elif _engine == 'CYCLES':
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
else:
    scene.render.engine = _engine
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = {out!r}

radius = 4.0
az = math.radians({azimuth})
el = math.radians({elevation})
cx = radius * math.sin(az) * math.cos(el)
cy = -radius * math.cos(az) * math.cos(el)
cz = radius * math.sin(el) + 0.5

for obj in [o for o in bpy.data.objects if o.name.startswith("RenderCam")]:
    bpy.data.objects.remove(obj, do_unlink=True)
for c in [c for c in bpy.data.cameras if c.name.startswith("RenderCam")]:
    bpy.data.cameras.remove(c)

cam = bpy.data.cameras.new("RenderCam")
cam_obj = bpy.data.objects.new("RenderCam", cam)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (cx, cy, cz)

direction = mathutils.Vector((0, 0, 0.5)) - mathutils.Vector((cx, cy, cz))
rot_quat = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot_quat.to_euler()

bpy.ops.render.render(write_still=True)
"""
            self._exec(code)
            paths.append(out)
        return paths

    def get_depth_map(self, output_path: str, resolution: tuple[int, int] = (512, 512)) -> str:
        code = f"""
import bpy
scene = bpy.context.scene
scene.render.resolution_x = {resolution[0]}
scene.render.resolution_y = {resolution[1]}
scene.render.use_compositing = True
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()
rl  = tree.nodes.new('CompositorNodeRLayers')
map_node = tree.nodes.new('CompositorNodeMapValue')
map_node.use_min = True; map_node.min[0] = 0.0
map_node.use_max = True; map_node.max[0] = 1.0
out = tree.nodes.new('CompositorNodeOutputFile')
out.base_path = ''
out.file_slots[0].path = {output_path!r}
tree.links.new(rl.outputs['Depth'],     map_node.inputs[0])
tree.links.new(map_node.outputs[0],     out.inputs[0])
scene.render.filepath = {output_path!r}
bpy.ops.render.render(write_still=True)
"""
        self._exec(code)
        return output_path

    def get_vertex_positions(self, object_name: str) -> list[list[float]]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj and obj.type == 'MESH':
    __result__ = [[round(v.co.x,4), round(v.co.y,4), round(v.co.z,4)] for v in obj.data.vertices[:2000]]
else:
    __result__ = []
""") or []

    def get_topology_stats(self, object_name: str) -> dict[str, Any]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj and obj.type == 'MESH':
    m = obj.data
    __result__ = {{"vertices": len(m.vertices), "edges": len(m.edges), "faces": len(m.polygons)}}
else:
    __result__ = {{}}
""") or {}

    @staticmethod
    def _parse_printed_result(result: Any, default: Any) -> Any:
        """Extract JSON from 'RESULT:<json>' in execute_code stdout."""
        if isinstance(result, (dict, list)):
            return result
        text = str(result or "")
        for line in text.splitlines():
            if line.startswith("RESULT:"):
                import json as _json
                try:
                    return _json.loads(line[7:])
                except Exception:
                    pass
        return default

    def apply_subdivision(self, object_name: str, levels: int = 2) -> dict[str, Any]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj:
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Subd", "SUBSURF")
    mod.levels = {levels}
    bpy.ops.object.modifier_apply(modifier=mod.name)
""")

    def smooth_mesh(self, object_name: str, iterations: int = 5, factor: float = 0.5) -> dict[str, Any]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj:
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Smooth", "SMOOTH")
    mod.iterations = {iterations}
    mod.factor = {factor}
    bpy.ops.object.modifier_apply(modifier=mod.name)
""")

    def unwrap_uv(self, object_name: str, method: str = "SMART_PROJECT") -> dict[str, Any]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    if {method!r} == 'SMART_PROJECT':
        bpy.ops.uv.smart_project()
    else:
        bpy.ops.uv.unwrap(method={method!r})
    bpy.ops.object.mode_set(mode='OBJECT')
""")

    def new_scene(self) -> dict[str, Any]:
        """Delete every object in the current scene, leaving a clean slate."""
        return self._exec("""
import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=True)
for block in list(bpy.data.meshes): bpy.data.meshes.remove(block)
for block in list(bpy.data.cameras): bpy.data.cameras.remove(block)
for block in list(bpy.data.lights): bpy.data.lights.remove(block)
""")

    def save_blend(self, path: str) -> dict[str, Any]:
        """Save the current .blend file to the given absolute path."""
        return self._exec(f"""
import bpy, os
os.makedirs(os.path.dirname({path!r}), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath={path!r})
""")

    def delete_object(self, object_name: str) -> dict[str, Any]:
        return self._exec(f"""
import bpy
obj = bpy.data.objects.get({object_name!r})
if obj:
    bpy.data.objects.remove(obj, do_unlink=True)
""")

    def clear_scene(self) -> dict[str, Any]:
        return self._exec(
            "import bpy; bpy.ops.wm.read_factory_settings(use_empty=True)"
        )
