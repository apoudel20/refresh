"""Coarse grid search over tongue bone angles/length (camera, head fixed); writes best into fit_params.json.
Usage: blender -b aux/renders/fitbase.blend -P aux/fit_tongue_grid.py"""
import sys, os, json, itertools, bpy
sys.argv = [sys.argv[0], "--", "1"]
src = open(os.path.join(os.path.dirname(os.path.abspath(bpy.data.filepath)), "..", "fit_pose.py")).read()
exec(src.split("# ---------------------------------------------------------------- coordinate descent")[0])
base = json.loads(json.dumps(bd.PARAMS))
best = (1e18, None)
for t1, t2, cu, sc_, jw in itertools.product((-25, -15, -5), (-10, 5, 20, 35), (-10, 10, 30), (0.8, 0.95, 1.1), (6, 14)):
    p = json.loads(json.dumps(base))
    p["pose"].update(tongue1=t1, tongue2=t2, tongue_curl=cu, tongue_scale=sc_, jaw=jw)
    f = evaluate(p)
    if f < best[0]:
        best = (f, p)
        print("[grid] new best", round(f), (t1, t2, cu, sc_, jw), flush=True)
f, d = evaluate(best[1], True)
print("[grid] final", round(f), "iou", d["tongue_iou"], d["lm"])
with open(os.path.join(HERE, "fit_params.json"), "w") as fh:
    json.dump(best[1], fh, indent=2)
