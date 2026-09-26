import bpy, sys
sc = bpy.context.scene
print("objects:")
for o in sorted(bpy.data.objects, key=lambda o: o.name):
    extra = ""
    if o.type == "ARMATURE":
        extra = f"bones={len(o.data.bones)} ik={sum(1 for pb in o.pose.bones for c in pb.constraints if c.type=='IK')}"
    if o.type == "MESH":
        extra = f"verts={len(o.data.vertices)} groups={len(o.vertex_groups)} mods={[m.type for m in o.modifiers]}"
    if o.type == "CURVES":
        extra = f"strands={len(o.data.curves)} surface={o.data.surface.name if o.data.surface else None} mods={[m.type for m in o.modifiers]}"
    print(f"  {o.name:22s} {o.type:9s} parent={o.parent.name if o.parent else '-':10s} {extra}")
print("camera", sc.camera.name, "engine", sc.render.engine, "samples", sc.cycles.samples, f"{sc.render.resolution_x}x{sc.render.resolution_y}")
# rig test: turn the head and make sure fur follows
rig = bpy.data.objects["Dog_Rig"]
fur = bpy.data.objects["Dog_Fur"]
dg = bpy.context.evaluated_depsgraph_get()
import numpy as np
def fur_pos():
    e = fur.evaluated_get(bpy.context.evaluated_depsgraph_get())
    a = np.empty(len(e.data.points) * 3, np.float32)
    e.data.attributes["position"].data.foreach_get("vector", a)
    return a.reshape(-1, 3)
p0 = fur_pos()
from mathutils import Quaternion
pb = rig.pose.bones["jaw"]; q = pb.rotation_quaternion.copy()
pb.rotation_quaternion = Quaternion((1, 0, 0), 0.3) @ q
bpy.context.view_layer.update()
p1 = fur_pos()
moved = np.linalg.norm(p1 - p0, axis=1) > 1e-4
print(f"jaw rotated -> {moved.sum()} of {len(p0)} fur points moved")
pb.rotation_quaternion = q
bpy.context.view_layer.update()
