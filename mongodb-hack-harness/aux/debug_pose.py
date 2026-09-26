import bpy, sys
rig = bpy.data.objects["Dog_Rig"]
for pb in rig.pose.bones:
    h = rig.matrix_world @ pb.head; t = rig.matrix_world @ pb.tail
    print(f"{pb.name:14s} head=({h.x:+.3f},{h.y:+.3f},{h.z:+.3f}) tail=({t.x:+.3f},{t.y:+.3f},{t.z:+.3f})")
