"""Fit the photo camera + head/neck pose so the model lines up with aux/reference.png.

Usage: blender -b aux/renders/fitbase.blend -P aux/fit_pose.py -- [max_evals]
Writes aux/fit_params.json (read by build_dog.py).

Objective = landmark reprojection error (eyes, nose, brow dot, mouth corner, tongue tip)
          + per-row silhouette error (left/right outline of the fur envelope vs. the reference mask).
The fur envelope is the deformed body surface pushed out along its normals by the local fur length.
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_dog as bd  # noqa: E402  (module import does not run main)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MAX_EVALS = int(argv[0]) if argv else 700
W, H = 740, 408

# ---------------------------------------------------------------- reference data
REF_LM = {  # pixel coords in the 740x408 reference, (x, y from top), weight
    "eye_near": ((292, 79), 1.5),
    "eye_far": ((358, 81), 0.6),
    "nose": ((391, 121), 1.5),
    "brow_near": ((302, 55), 0.8),
    "mouth_corner": ((277, 184), 1.0),
    "tongue_tip": ((414, 228), 1.0),
    "tongue_exit": ((336, 196), 1.0),
    "chin": ((345, 222), 0.3),
}
ref_img = bpy.data.images.load(os.path.join(HERE, "reference.png"))
ref = np.array(ref_img.pixels[:], np.float32).reshape(H, W, 4)[::-1, :, :3]
r, g, b = ref[..., 0], ref[..., 1], ref[..., 2]
ref_mask = ~((b - r > 0.22) & (g > 0.45) & (b > 0.6))
ROWS = np.arange(36, H, 6)


def extents(mask):
    L = np.full(len(ROWS), np.nan)
    Rr = np.full(len(ROWS), np.nan)
    for k, y in enumerate(ROWS):
        xs = np.nonzero(mask[y])[0]
        if len(xs):
            L[k], Rr[k] = xs.min(), xs.max()
    return L, Rr


REF_L, REF_R = extents(ref_mask)
# tongue target: pink pixels below the upper lip line (excludes gums/palate inside the mouth)
_yy = np.arange(H)[:, None] + np.zeros((1, W))
REF_TONGUE = (r - g > 0.18) & (b - g > 0.0) & (_yy > 186) & (np.arange(W)[None, :] > 300)
REF_TOP = int(np.nonzero(ref_mask.any(1))[0][0])

# ---------------------------------------------------------------- scene handles
sc = bpy.context.scene
rig = bpy.data.objects["Dog_Rig"]
body = bpy.data.objects["Dog_Body"]
tongue = bpy.data.objects["Dog_Tongue"]
nose = bpy.data.objects["Dog_Nose"]
cam = sc.camera
fur = bpy.data.objects["Dog_Fur"]


def eval_fur():
    e = fur.evaluated_get(bpy.context.evaluated_depsgraph_get())
    n = len(e.data.points)
    a = np.empty(n * 3, np.float32)
    e.data.attributes["position"].data.foreach_get("vector", a)
    mw = np.array(fur.matrix_world)
    return a.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]


def bone_point(bone, rest):
    pb = rig.pose.bones[bone]
    m = rig.matrix_world @ pb.matrix @ rig.data.bones[bone].matrix_local.inverted()
    return np.array(m @ Vector(rest))


def head_point(rest):
    pb = rig.pose.bones["head"]
    m = rig.matrix_world @ pb.matrix @ rig.data.bones["head"].matrix_local.inverted()
    return np.array(m @ Vector(rest))


def eval_verts(obj, with_normals=False):
    dg = bpy.context.evaluated_depsgraph_get()
    e = obj.evaluated_get(dg)
    me = e.to_mesh()
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    nor = None
    if with_normals:
        nor = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("normal", nor)
        nor = nor.reshape(-1, 3).astype(np.float64)
    mw = np.array(obj.matrix_world)
    co = co @ mw[:3, :3].T + mw[:3, 3]
    if nor is not None:
        nor = nor @ mw[:3, :3].T
    e.to_mesh_clear()
    return co, nor


def project(pts):
    """World points -> pixel coords (x, y from top) for the current camera, matching Cycles framing."""
    mv = np.array(cam.matrix_world.inverted())
    pc = pts @ mv[:3, :3].T + mv[:3, 3]
    z = np.where(-pc[:, 2] > 1e-3, -pc[:, 2], np.nan)
    k = cam.data.lens / cam.data.sensor_width
    x = W * (0.5 + k * pc[:, 0] / z - cam.data.shift_x)
    y_up = H * 0.5 + W * (k * pc[:, 1] / z - cam.data.shift_y)
    return np.stack([x, H - y_up], 1)


def silhouette(pix):
    m = np.zeros((H, W), bool)
    xi = np.round(pix[:, 0]).astype(int)
    yi = np.round(pix[:, 1]).astype(int)
    ok = np.isfinite(pix).all(1)
    pix = np.nan_to_num(pix, nan=-1)
    xi = np.round(pix[:, 0]).astype(int)
    yi = np.round(pix[:, 1]).astype(int)
    ok &= (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    m[yi[ok], xi[ok]] = True
    # close small gaps between projected points
    for _ in range(2):
        d = m.copy()
        d[1:] |= m[:-1]
        d[:-1] |= m[1:]
        d[:, 1:] |= m[:, :-1]
        d[:, :-1] |= m[:, 1:]
        m = d
    return m


# commissure = rearmost lip vertex on the camera side (rest mesh), tracked through the deformation
_co = np.empty(len(body.data.vertices) * 3, np.float32)
body.data.vertices.foreach_get("co", _co)
_co = _co.reshape(-1, 3)
_wet = np.empty(len(body.data.vertices), np.float32)
body.data.attributes["wet"].data.foreach_get("value", _wet)
_cand = np.nonzero((_wet > 0.3) & (_co[:, 1] < -0.02) & (_co[:, 0] > 0.4))[0]
CORNER_V = int(_cand[np.argmin(_co[_cand, 0])])
print("[fit] commissure vertex", CORNER_V, _co[CORNER_V])
_last = {}


def mouth_corner():
    return _last["body"][CORNER_V]


# ---------------------------------------------------------------- objective
state = {"spine": None}


def apply(params):
    pp, cp = params["pose"], params["cam"]
    if state["spine"] != pp["spine"]:
        bd.apply_photo_pose(rig, pp)
        state["spine"] = pp["spine"]
    else:
        bd.apply_head_pose(rig, pp)
    bd.place_photo_camera(cam, cp)
    bpy.context.view_layer.update()


def evaluate(params, detail=False):
    apply(params)
    env, _ = eval_verts(body)
    _last["body"] = env
    env = np.concatenate([env, eval_fur()])
    tco, _ = eval_verts(tongue)
    # keep only tongue vertices the camera can see past the lips/jaw
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    cpos = cam.matrix_world.translation
    vis = np.zeros(len(tco), bool)
    for i, p in enumerate(tco):
        d = Vector(p) - cpos
        L = d.length
        vis[i] = bvh.ray_cast(cpos, d / L, L - 0.002)[0] is None
    tco = tco[vis]
    tpix = project(tco)
    tm = silhouette(tpix) & (_yy > 186) & (np.arange(W)[None, :] > 300)
    inter = (tm & REF_TONGUE).sum()
    union = max((tm | REF_TONGUE).sum(), 1)
    tongue_iou = inter / union
    nco, _ = eval_verts(nose)
    pts = np.concatenate([env, tco, nco])
    sil = silhouette(project(pts))
    L, Rr = extents(sil)
    lm = {
        "eye_near": np.array(bpy.data.objects["Dog_Eye.R"].matrix_world.translation),
        "eye_far": np.array(bpy.data.objects["Dog_Eye.L"].matrix_world.translation),
        "nose": np.array(nose.matrix_world.translation),
        "brow_near": head_point((0.500, -0.040, 0.924)),
        "mouth_corner": mouth_corner(),
        "tongue_tip": np.array(rig.matrix_world @ rig.pose.bones["tongue_03"].tail),
        "tongue_exit": np.array(rig.matrix_world @ rig.pose.bones["tongue_02"].head),
        "chin": bone_point("jaw", bd.jaw((0.605, -0.01, 0.815))),
    }
    names = list(lm)
    pix = project(np.array([lm[n] for n in names]))
    lm_err = 0.0
    per = {}
    for n, p in zip(names, pix):
        (tx, ty), w = REF_LM[n]
        e = math.hypot(p[0] - tx, p[1] - ty)
        per[n] = (round(float(p[0]), 1), round(float(p[1]), 1), round(e, 1))
        lm_err += w * e * e
    dl = np.nan_to_num(L - REF_L, nan=80.0)
    dr = np.nan_to_num(Rr - REF_R, nan=80.0)
    # the jaw/tongue rows are noisy (mouth gaps), trust the outline less there
    wrow = np.where((ROWS > 160) & (ROWS < 250), 0.3, 1.0)
    sil_err = float((wrow * (np.minimum(np.abs(dl), 80) ** 2 + np.minimum(np.abs(dr), 80) ** 2)).sum())
    top = int(np.nonzero(sil.any(1))[0][0]) if sil.any() else 0
    top_err = (top - REF_TOP) ** 2
    total = lm_err + 0.25 * sil_err + 2.0 * top_err + 4000.0 * (1 - tongue_iou)
    if detail:
        return total, dict(tongue_iou=round(float(tongue_iou), 3), lm=per, top=top, lm_err=lm_err, sil_err=sil_err, dl=dl, dr=dr)
    return total


# ---------------------------------------------------------------- coordinate descent
params = json.loads(json.dumps(bd.PARAMS))
params["pose"].setdefault("tongue1", 0.0)
params["pose"].setdefault("tongue2", -4.0)
params["pose"].setdefault("tongue_scale", 1.0)
STEPS = {
    ("cam", "az"): 3.0, ("cam", "el"): 3.0, ("cam", "dist"): 0.08, ("cam", "roll"): 2.0,
    ("cam", "sx"): 0.02, ("cam", "sy"): 0.02,
    ("pose", "spine"): 4.0, ("pose", "neck1"): 4.0, ("pose", "neck2"): 4.0, ("pose", "neck_yaw"): 4.0,
    ("pose", "head_pitch"): 4.0, ("pose", "head_yaw"): 4.0, ("pose", "head_roll"): 3.0,
    ("pose", "jaw"): 3.0, ("pose", "tongue_curl"): 6.0, ("pose", "tongue_roll"): 8.0,
    ("pose", "tongue1"): 5.0, ("pose", "tongue2"): 5.0, ("pose", "tongue_scale"): 0.08,
}
if os.environ.get("FIT_KEYS"):
    keep = {tuple(k.split(".")) for k in os.environ["FIT_KEYS"].split(",")}
    STEPS = {k: v for k, v in STEPS.items() if k in keep}
LIMITS = {("pose", "tongue_curl"): (-10, 45), ("pose", "tongue_roll"): (0, 35), ("pose", "tongue1"): (-30, 20),
          ("pose", "tongue2"): (-40, 20), ("pose", "tongue_scale"): (0.6, 1.3), ("pose", "jaw"): (-10, 14), ("pose", "spine"): (10, 45), ("cam", "dist"): (1.2, 3.5)}
t0 = time.time()


def nelder_mead(params, keys, max_evals):
    """Simplex search over the given (group, name) keys; handles coupled parameters."""
    def to_params(v):
        p = json.loads(json.dumps(params))
        for (grp, name), x in zip(keys, v):
            lo, hi = LIMITS.get((grp, name), (-1e9, 1e9))
            p[grp][name] = float(min(max(x, lo), hi))
        return p
    x0 = np.array([params[g][n] for g, n in keys], float)
    simplex = [x0] + [x0 + np.eye(len(keys))[i] * STEPS[k] * 1.5 for i, k in enumerate(keys)]
    fs = [evaluate(to_params(x)) for x in simplex]
    n_ev = len(fs)
    while n_ev < max_evals:
        order = np.argsort(fs)
        simplex = [simplex[i] for i in order]
        fs = [fs[i] for i in order]
        if n_ev % 20 < len(keys) + 2:
            print(f"[fit] nm evals {n_ev} best {fs[0]:.0f}", flush=True)
        c = np.mean(simplex[:-1], 0)
        xr = c + (c - simplex[-1]); fr = evaluate(to_params(xr)); n_ev += 1
        if fr < fs[0]:
            xe = c + 2 * (c - simplex[-1]); fe = evaluate(to_params(xe)); n_ev += 1
            simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            simplex[-1], fs[-1] = xr, fr
        else:
            xc = c + 0.5 * (simplex[-1] - c); fc = evaluate(to_params(xc)); n_ev += 1
            if fc < fs[-1]:
                simplex[-1], fs[-1] = xc, fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fs[i] = evaluate(to_params(simplex[i])); n_ev += 1
        if np.std(fs) < 1.0:
            break
    i = int(np.argmin(fs))
    return to_params(simplex[i]), fs[i]


if os.environ.get("FIT_METHOD") == "nm":
    params, _ = nelder_mead(params, list(STEPS), MAX_EVALS)
    MAX_EVALS = 0
best = evaluate(params)
_, d0 = evaluate(params, True)
print(f"[fit] start {best:.0f}  lm {d0['lm_err']:.0f} sil {d0['sil_err']:.0f} top {d0['top']}  {d0['lm']}", flush=True)
evals = 2
steps = dict(STEPS)
while evals < MAX_EVALS and max(steps[k] / STEPS[k] for k in steps) > 0.06:
    improved = False
    for key in STEPS:
        grp, name = key
        for sgn in (1, -1):
            trial = json.loads(json.dumps(params))
            v = trial[grp][name] + sgn * steps[key]
            lo, hi = LIMITS.get(key, (-1e9, 1e9))
            trial[grp][name] = min(max(v, lo), hi)
            f = evaluate(trial)
            evals += 1
            if f < best:
                best, params, improved = f, trial, True
                steps[key] *= 1.3
                break
        else:
            steps[key] *= 0.6
    print(f"[fit] evals {evals} best {best:.0f} ({time.time() - t0:.0f}s)", flush=True)
    if not improved and all(steps[k] / STEPS[k] < 0.06 for k in steps):
        break

final, d = evaluate(params, True)
print(f"[fit] final {final:.0f}  tongue IoU {d['tongue_iou']}  lm {d['lm_err']:.0f} sil {d['sil_err']:.0f} top {d['top']} (ref {REF_TOP})")
for n, v in d["lm"].items():
    print(f"[fit]   {n:13s} model {v[:2]} ref {REF_LM[n][0]} err {v[2]}px")
print("[fit] row  dL   dR")
for y, a, b in zip(ROWS[::4], d["dl"][::4], d["dr"][::4]):
    print(f"[fit] {y:4d} {a:+5.0f} {b:+5.0f}")
out = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in params.items()}
with open(os.path.join(HERE, "fit_params.json"), "w") as f:
    json.dump(out, f, indent=2)
print("[fit] params", json.dumps(out))
