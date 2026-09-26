"""Procedurally builds a rigged, furred Bernese Mountain Dog matching aux/reference.png.

Run:
  blender -b --factory-startup -P aux/build_dog.py -- [--nofur] [--render PATH]
         [--cam photo|side|front|close] [--samples N] [--pct P] [--save PATH]

Dog rest pose: facing +X, Z up, +Y is the dog's left side, units in meters.
"""
import bpy
import math
import os
import sys
import time

import numpy as np
from mathutils import Euler, Matrix, Vector
from mathutils.kdtree import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# --------------------------------------------------------------------------- args
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {"fur": True, "render": None, "cam": "photo", "samples": 64, "pct": 50,
            "save": os.path.join(ROOT, "outputs", "bernese_dog.blend"), "strands": 450000,
            "pose": True, "border": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--nofur":
            opts["fur"] = False
        elif a == "--nopose":
            opts["pose"] = False
        elif a == "--nosave":
            opts["save"] = None
        elif a in ("--render", "--cam", "--save"):
            opts[a[2:]] = argv[i + 1]
            i += 1
        elif a == "--border":
            opts["border"] = [float(v) for v in argv[i + 1].split(",")]
            i += 1
        elif a in ("--samples", "--pct", "--strands"):
            opts[a[2:]] = int(argv[i + 1])
            i += 1
        i += 1
    return opts


OPTS = parse_args()

# Photo-matching camera + pose parameters. aux/fit_pose.py optimises these against the reference
# and writes aux/fit_params.json, which overrides the defaults below.
PARAMS = {
    "cam": {"tx": 0.27, "ty": 0.0, "tz": 0.85, "az": 40.07, "el": -1.16, "dist": 1.973, "roll": 0.0,
            "sx": 0.175, "sy": -0.182, "lens": 85.0},
    "pose": {"spine": 30.0, "neck1": -16.0, "neck2": -8.0, "neck_yaw": 2.0, "head_pitch": -16.0, "head_yaw": 4.0,
             "head_roll": -5.0, "jaw": 0.0, "tongue_curl": 16.0, "tongue_roll": 22.0, "tongue1": 0.0,
             "tongue2": -4.0, "tongue_scale": 1.0},
}
_FIT = os.path.join(HERE, "fit_params.json")
if os.path.exists(_FIT) and not os.environ.get("DOG_NO_FIT"):
    import json
    with open(_FIT) as f:
        _fp = json.load(f)
    for k in PARAMS:
        PARAMS[k].update(_fp.get(k, {}))
RNG = np.random.default_rng(7)


def log(*a):
    print("[dog]", *a, flush=True)


# --------------------------------------------------------------------------- scene
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    return scene


def link(obj, coll=None):
    (coll or bpy.context.scene.collection).objects.link(obj)
    return obj


def new_collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


# --------------------------------------------------------------------------- metaball body
THRESH = 0.6


def kfac(s):
    return math.sqrt(1.0 - (THRESH / s) ** (1.0 / 3.0))


def rot_about(p, pivot, angle_y):
    """Rotate point p about pivot around the Y axis (positive tips +X down)."""
    m = Matrix.Rotation(angle_y, 3, "Y")
    return tuple(Vector(pivot) + m @ (Vector(p) - Vector(pivot)))


MOUTH_NEG = []  # (center, semi, rotation-matrix, stiffness, radius) of the mouth carving elements
TAGGED = {"jaw": [], "upper": []}  # positive ellipsoids by region, same tuple layout


class Meta:
    def __init__(self, name):
        self.mb = bpy.data.metaballs.new(name)
        self.mb.threshold = THRESH
        self.tag = None

    def ell(self, c, semi, rot=(0, 0, 0), s=3.0, neg=False):
        el = self.mb.elements.new(type="ELLIPSOID")
        el.co = c
        el.stiffness = s
        el.radius = 1.0 / kfac(s)
        el.size_x, el.size_y, el.size_z = semi
        el.rotation = Euler(rot).to_quaternion()
        el.use_negative = neg
        rec = (np.array(c), np.array(semi), np.array(Euler(rot).to_matrix()), s, el.radius)
        if neg and self.tag == "mouth":
            MOUTH_NEG.append(rec)
        elif not neg and self.tag in TAGGED:
            TAGGED[self.tag].append(rec)
        return el

    def cap(self, a, b, r, s=3.0, neg=False):
        a, b = Vector(a), Vector(b)
        d = b - a
        el = self.mb.elements.new(type="CAPSULE")
        el.co = (a + b) / 2
        el.size_x = d.length / 2
        el.stiffness = s
        el.radius = r / kfac(s)
        el.rotation = Vector((1, 0, 0)).rotation_difference(d.normalized())
        el.use_negative = neg
        return el

    def ball(self, c, r, s=3.0, neg=False):
        return self.ell(c, (r, r, r), s=s, neg=neg)


R = math.radians
JAW_PIVOT = (0.462, 0.0, 0.822)
JAW_OPEN = R(34)
EYE_R = 0.0115
EYES = [(0.508, 0.046 * sy, 0.898) for sy in (1, -1)]
NOSE_TIP = (0.652, 0.0, 0.852)


def jaw(p):
    return rot_about(p, JAW_PIVOT, JAW_OPEN)


# tongue centreline: lies on the floor of the (closed) jaw, then spills over the lower lip
TONGUE_SPINE = [jaw(p) for p in [(0.475, 0, 0.824), (0.52, 0, 0.8255), (0.56, 0, 0.8275), (0.60, 0.002, 0.8295),
                                  (0.62, 0.005, 0.829), (0.638, 0.008, 0.826), (0.652, 0.010, 0.820),
                                  (0.658, 0.010, 0.812)]]


def build_body_meta():
    m = Meta("DogMeta")
    E, C = m.ell, m.cap
    # ---- torso (deep chest, level topline, slight tuck)
    E((0.19, 0, 0.53), (0.16, 0.145, 0.165), s=3)
    E((0.00, 0, 0.545), (0.21, 0.148, 0.155), s=3)
    E((-0.21, 0, 0.56), (0.15, 0.13, 0.125), s=3)
    E((-0.35, 0, 0.575), (0.12, 0.135, 0.12), s=3)
    E((0.29, 0, 0.50), (0.08, 0.115, 0.12), s=3)  # prosternum / chest front
    for sy in (1, -1):
        E((0.21, 0.085 * sy, 0.56), (0.10, 0.06, 0.13), rot=(0, R(-15), 0), s=4)  # shoulder
    # ---- neck (thick, ruffed)
    E((0.29, 0, 0.665), (0.125, 0.12, 0.12), rot=(0, R(-45), 0), s=3)
    E((0.365, 0, 0.765), (0.09, 0.10, 0.105), rot=(0, R(-30), 0), s=3)
    E((0.385, 0, 0.685), (0.065, 0.085, 0.085), s=4)  # throat / front of neck
    # ---- head
    E((0.425, 0, 0.872), (0.09, 0.078, 0.066), s=4)  # cranium
    E((0.395, 0, 0.855), (0.065, 0.072, 0.07), s=4)  # occiput
    for sy in (1, -1):
        E((0.468, 0.048 * sy, 0.838), (0.058, 0.04, 0.046), s=5)  # cheek / masseter
        E((0.498, 0.034 * sy, 0.906), (0.026, 0.024, 0.016), s=7)  # brow
    m.tag = "upper"
    E((0.572, 0, 0.850), (0.074, 0.045, 0.036), rot=(0, R(4), 0), s=6)  # muzzle bridge
    E((0.600, 0, 0.808), (0.040, 0.030, 0.026), s=7)  # front of the upper lip under the nose
    E((0.52, 0, 0.872), (0.03, 0.03, 0.03), s=6)  # stop fill
    E((0.492, 0, 0.893), (0.042, 0.05, 0.03), s=5)  # forehead between the brows
    for sy in (1, -1):
        E((0.562, 0.031 * sy, 0.810), (0.068, 0.024, 0.032), s=7)  # upper lip / flews (deep, pendulous)
        E((0.49, 0.042 * sy, 0.800), (0.032, 0.02, 0.024), s=7)  # mouth corner
        E((0.515, 0.037 * sy, 0.806), (0.036, 0.018, 0.022), s=7)  # hanging flews at the back
    # ---- lower jaw (opened)
    m.tag = "jaw"
    E(jaw((0.535, 0, 0.800)), (0.066, 0.036, 0.020), rot=(0, JAW_OPEN, 0), s=7)
    E(jaw((0.585, 0, 0.806)), (0.026, 0.026, 0.012), rot=(0, JAW_OPEN, 0), s=7)  # chin
    for sy in (1, -1):
        E(jaw((0.52, 0.028 * sy, 0.81)), (0.055, 0.012, 0.012), rot=(0, JAW_OPEN, 0), s=8)  # lower lip
    # mouth cavity carve
    m.tag = "mouth"
    E(jaw((0.555, 0, 0.812)), (0.09, 0.031, 0.014), rot=(0, JAW_OPEN * 0.5, 0), s=7, neg=True)
    E((0.585, 0, 0.800), (0.075, 0.029, 0.013), s=7, neg=True)
    for sy in (1, -1):
        E(rot_about((0.505, 0.034 * sy, 0.802), JAW_PIVOT, JAW_OPEN * 0.5), (0.045, 0.013, 0.011),
          rot=(0, JAW_OPEN * 0.5, 0), s=7, neg=True)  # open the commissures
    m.tag = None
    # eye sockets
    for c in EYES:
        m.ball(c, EYE_R * 0.9, s=8, neg=True)
    # ---- ears (flat, hanging, rolled against head)
    for sy in (1, -1):
        E((0.405, 0.086 * sy, 0.842), (0.052, 0.015, 0.078), rot=(R(-26) * sy, R(-10), 0), s=7)
        E((0.405, 0.080 * sy, 0.905), (0.035, 0.02, 0.02), s=6)  # ear base
    # ---- front legs
    for sy in (1, -1):
        y = 0.10 * sy
        C((0.22, y, 0.49), (0.185, y, 0.31), 0.058, s=4)
        C((0.185, y, 0.31), (0.195, y, 0.07), 0.044, s=5)
        E((0.22, y, 0.032), (0.06, 0.044, 0.03), s=6)
    # ---- hind legs
    for sy in (1, -1):
        y = 0.095 * sy
        E((-0.34, y, 0.47), (0.11, 0.065, 0.15), rot=(0, R(15), 0), s=4)
        C((-0.31, y, 0.34), (-0.43, y, 0.16), 0.047, s=5)
        C((-0.43, y, 0.16), (-0.41, y, 0.06), 0.037, s=5)
        E((-0.37, y, 0.03), (0.06, 0.042, 0.03), s=6)
    # ---- tail
    C((-0.46, 0, 0.61), (-0.535, 0, 0.51), 0.034, s=4)
    C((-0.535, 0, 0.51), (-0.555, 0, 0.37), 0.028, s=5)
    C((-0.555, 0, 0.37), (-0.53, 0, 0.23), 0.02, s=5)
    return m.mb


def eval_to_mesh(obj, name):
    dg = bpy.context.evaluated_depsgraph_get()
    return bpy.data.meshes.new_from_object(obj.evaluated_get(dg), preserve_all_data_layers=True,
                                           depsgraph=dg)


def build_body(coll):
    t = time.time()
    mb = build_body_meta()
    mb.resolution = 0.006
    mb.render_resolution = 0.006
    mobj = link(bpy.data.objects.new("DogMeta", mb), coll)
    me = eval_to_mesh(mobj, "DogBody")
    bpy.data.objects.remove(mobj)
    bpy.data.metaballs.remove(mb)
    body = link(bpy.data.objects.new("Dog_Body", me), coll)
    me.name = "Dog_Body"
    # clean, even topology then a gentle smooth to blend junctions
    rm = body.modifiers.new("Remesh", "REMESH")
    rm.mode = "VOXEL"
    rm.voxel_size = 0.0045
    sm = body.modifiers.new("Smooth", "LAPLACIANSMOOTH")
    sm.lambda_factor = 0.6
    sm.iterations = 4
    sm.use_volume_preserve = True
    me2 = eval_to_mesh(body, "DogBody")
    body.modifiers.clear()
    body.data = me2
    bpy.data.meshes.remove(me)
    me2.name = "Dog_Body"
    for p in me2.polygons:
        p.use_smooth = True
    log(f"body mesh {len(me2.vertices)} verts in {time.time() - t:.1f}s")
    return body


# --------------------------------------------------------------------------- coat maps
def ellv(p, c, semi):
    d = p - np.array(c, dtype=float)
    return sum((d[:, i] / semi[i]) ** 2 for i in range(3))


def smooth01(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def fbm_noise(p, freq, seed=0, octaves=6):
    """Cheap smooth-ish 3D noise from sums of sines (vectorized, deterministic)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(p))
    for i in range(octaves):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        ph = rng.uniform(0, 6.28)
        f = freq * (1.7 ** i)
        out += np.sin(p @ d * f + ph) / (1.5 ** i)
    return out / 2.0


BLACK = np.array([0.006, 0.0055, 0.0055])
WHITE = np.array([0.85, 0.83, 0.78])
TAN = np.array([0.30, 0.075, 0.018])


def elem_field(p, recs):
    f = np.zeros(len(p))
    for c, semi, rm, st, rad in recs:
        local = (p - c) @ rm
        d2 = ((local / semi) ** 2).sum(1)
        f += st * np.clip(1 - d2 / (rad * rad), 0, None) ** 3
    return f


def jawness(p):
    """1 for lower-jaw vertices, 0 for the rest of the head (by metaball element ownership)."""
    fj, fu = elem_field(p, TAGGED["jaw"]), elem_field(p, TAGGED["upper"])
    return smooth01(-0.15, 0.15, fj - fu) * (fj > 1e-4)


def column_band(p, mask, use_min, width, bx=0.003, by=0.006):
    """Band of `width` along the lowest (use_min) / highest surface of the masked verts, per x/|y| column."""
    out = np.zeros(len(p))
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return out
    q = p[idx]
    key = (np.floor(q[:, 0] / bx).astype(np.int64) * 1000 + np.floor(np.abs(q[:, 1]) / by).astype(np.int64))
    order = np.argsort(key)
    ks, qs = key[order], q[order, 2]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    red = np.minimum.reduceat(qs, starts) if use_min else np.maximum.reduceat(qs, starts)
    ext = np.repeat(red, np.diff(np.r_[starts, len(ks)]))
    dist = (qs - ext) if use_min else (ext - qs)
    vals = smooth01(width, width * 0.4, dist)
    out[idx[order]] = vals
    return out


def smooth_scalar(edges, vals, iters=4):
    """Average a per-vertex scalar with its mesh neighbours."""
    v = vals.astype(float).copy()
    a, b = edges[:, 0], edges[:, 1]
    deg = np.bincount(a, minlength=len(v)) + np.bincount(b, minlength=len(v))
    for _ in range(iters):
        acc = np.bincount(a, weights=v[b], minlength=len(v)) + np.bincount(b, weights=v[a], minlength=len(v))
        v = 0.5 * v + 0.5 * acc / np.maximum(deg, 1)
    return v


def coat_maps(p, n, edges):
    """Returns (color, fur_length, fur_density, flow_dir, wet, skin_tint) for rest positions."""
    N = len(p)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    ay = np.abs(y)
    nz = fbm_noise(p, 60.0, 1) * 0.5 + fbm_noise(p, 180.0, 2) * 0.25

    head_e = np.minimum(ellv(p, (0.44, 0, 0.865), (0.115, 0.11, 0.095)),
                        ellv(p, (0.555, 0, 0.805), (0.12, 0.075, 0.10)))
    head = smooth01(1.25, 0.95, head_e)

    # ---------------- white (0..1)
    # blaze: narrow between eyes, widening towards stop and top of skull
    blaze_w = 0.005 + 0.007 * smooth01(0.44, 0.52, x) + 0.016 * smooth01(0.52, 0.56, x)
    blaze_w = np.where(z > 0.886, np.minimum(blaze_w, 0.006 + 0.004 * smooth01(0.50, 0.54, x)), blaze_w)
    blaze = (ay < blaze_w + nz * 0.003) & (z > 0.86) & (x > 0.495)
    white = blaze.astype(float) * head
    # muzzle: white in front of a line running from the stop down/back to the mouth corner
    muzzle_edge = 0.530 + 0.33 * np.clip(0.885 - z, 0, 0.2) + nz * 0.008
    jw_c = jawness(p)
    muzzle_top = smooth01(0.892, 0.884, z + nz * 0.004)  # muzzle sides stay below eye level
    white = np.maximum(white, smooth01(-0.004, 0.004, x - muzzle_edge) * head * (z > 0.70) * (1 - jw_c) * muzzle_top)
    # chin / underside of jaw
    jawp = np.array(JAW_PIVOT)
    white = np.maximum(white, 0.5 * smooth01(1.2, 0.7, ellv(p, jaw((0.60, 0, 0.78)), (0.03, 0.025, 0.02)) + nz * 0.3))  # chin
    # throat: pale fur from under the jaw down into the chest blaze
    throat = smooth01(1.3, 0.8, ellv(p, (0.44, 0, 0.745), (0.085, 0.055, 0.06)) + nz * 0.4) * (n[:, 2] < 0.3)
    white = np.maximum(white, throat * 0.85)
    # chest cross: front of throat and chest
    fwd = n[:, 0] * 0.9 - n[:, 2] * 0.2
    chest = smooth01(1.35, 0.75, ellv(p, (0.36, 0, 0.58), (0.14, 0.048 + 0.03 * np.clip(0.72 - z, 0, 1), 0.21)) + nz * 0.35)
    chest *= (fwd > -0.1) & (z < 0.80) & (z > 0.33)
    white = np.maximum(white, chest * 0.97)
    # paws, tail tip
    white = np.maximum(white, smooth01(0.075, 0.055, z + nz * 0.01) * (np.abs(x) > 0.1))
    white = np.maximum(white, (ellv(p, (-0.53, 0, 0.22), (0.05, 0.05, 0.05)) < 1.0) * 1.0)

    # ---------------- tan (0..1)
    tan = np.zeros(N)
    for sy in (1, -1):
        tan = np.maximum(tan, smooth01(1.2, 0.7, ellv(p, (0.500, 0.040 * sy, 0.924), (0.014, 0.016, 0.010))))
        tan = np.maximum(tan, smooth01(1.2, 0.8, ellv(p, (0.49, 0.056 * sy, 0.815), (0.05, 0.036, 0.058)) + nz * 0.3))
        tan = np.maximum(tan, jw_c * smooth01(1.3, 0.8, ellv(p, jaw((0.53, 0.03 * sy, 0.79)), (0.07, 0.035, 0.035))))
    # border around chest white
    ring = (ellv(p, (0.36, 0, 0.60), (0.15, 0.065, 0.24)) < 1.0 + nz * 0.3) & (fwd > -0.3) & (z < 0.78) & (z > 0.33)
    tan = np.maximum(tan, ring * 0.4 * smooth01(0.72, 0.77, z))
    # legs
    front_face = smooth01(-0.35, 0.1, n[:, 0])
    fl = (z < 0.26) & (z > 0.05) & (x > 0.1)
    tan = np.maximum(tan, fl * smooth01(0.24, 0.19, z + nz * 0.02) * front_face)
    inner_leg = smooth01(0.0, 0.5, -n[:, 1] * np.sign(y))  # inside of the legs
    hl = (z < 0.36) & (z > 0.05) & (x < -0.25)
    tan = np.maximum(tan, hl * smooth01(0.30, 0.22, z + nz * 0.02) * np.maximum(front_face, inner_leg))

    col = BLACK[None, :] * np.ones((N, 1))
    col = col + (TAN - BLACK)[None, :] * tan[:, None]
    col = col + (WHITE - col) * white[:, None]
    col = col * (1 - 0.08 * chest[:, None] * (1 - head[:, None]))  # chest bib slightly greyer
    # subtle variation
    col *= (1.0 + 0.12 * nz)[:, None]

    # ---------------- mouth interior, lips, eyes, nose masks
    # metaball field of the mouth carving elements: >0 only near the carved surface
    negf = np.zeros(N)
    for c, semi, rm, st, rad in MOUTH_NEG:
        local = (p - c) @ rm  # into element frame
        d2 = ((local / semi) ** 2).sum(1)
        t = np.clip(1 - d2 / (rad * rad), 0, None)
        negf += st * t ** 3
    if os.environ.get("DOG_DEBUG"):
        mz = (x > 0.45) & (z > 0.72) & (z < 0.88) & (ay < 0.07)
        log("negf pct in mouth zone:", np.percentile(negf[mz], [50, 75, 90, 95, 99]).round(3), "max", negf.max().round(3))
        log("negf>0 count", int((negf > 0).sum()), "zone", int(mz.sum()))
    inner = smooth01(0.35, 0.8, negf)
    lip = smooth01(0.05, 0.3, negf) * (1 - inner)
    eye_d = np.min([np.linalg.norm(p - np.array(c), axis=1) for c in EYES], axis=0)
    eyelid = smooth01(EYE_R * 1.7, EYE_R * 1.15, eye_d)
    nose_d = np.linalg.norm((p - np.array(NOSE_TIP)) / np.array([1.0, 1.0, 0.85]), axis=1)
    nosem = smooth01(0.034, 0.024, nose_d)

    skin = col.copy()
    pink = np.array([0.13, 0.02, 0.025])
    skin = skin * (1 - inner[:, None]) + pink[None, :] * inner[:, None]
    skin = skin * (1 - lip[:, None]) + np.array([0.02, 0.012, 0.012])[None, :] * lip[:, None]
    skin = skin * (1 - eyelid[:, None]) + np.array([0.01, 0.008, 0.008])[None, :] * eyelid[:, None]
    wet = np.clip(inner + lip * 0.6 + eyelid * 0.5, 0, 1)

    # ---------------- fur length (m)
    L = np.full(N, 0.045)
    L = np.where(z > 0.55, 0.05, L)
    neck = smooth01(0.22, 0.28, x) * smooth01(0.44, 0.40, x) * smooth01(0.55, 0.65, z)
    L = L + neck * (0.06 + 0.03 * smooth01(0.70, 0.80, z)) * (1 - 0.5 * smooth01(0.3, 0.9, n[:, 0]))
    L = L * (1 - head) + head * (0.013 + 0.012 * smooth01(0.43, 0.38, x))
    face = head * smooth01(0.47, 0.51, x)
    L = L * (1 - face) + face * 0.0055
    L = np.where(eye_d < EYE_R * 2.4, np.minimum(L, 0.0035), L)
    # ears: medium fur with long fringe lower & back
    for sy in (1, -1):
        ear = smooth01(1.3, 0.9, ellv(p, (0.412, 0.093 * sy, 0.853), (0.05, 0.03, 0.075)))
        ear_len = 0.04 + 0.05 * smooth01(0.87, 0.79, z) + 0.02 * smooth01(0.42, 0.37, x)
        L = L * (1 - ear) + ear * ear_len
    legs = smooth01(0.34, 0.26, z)
    feather = legs * (x > 0.1) * smooth01(0.0, -0.6, n[:, 0])  # back of forelegs
    L = L * (1 - legs) + legs * (0.014 + 0.035 * feather)
    L = np.where(z < 0.07, 0.008, L)
    belly = smooth01(-0.2, -0.8, n[:, 2]) * smooth01(0.30, 0.42, z) * (x > -0.3) * (x < 0.28)
    L = L + belly * 0.035
    tail = (x < -0.45) & (z < 0.62)
    L = np.where(tail, 0.085, L)
    chest_front = chest * 1.0
    L = L + chest_front * 0.045 * (1 - head)
    L *= 1.0 + 0.15 * nz

    # ---------------- density multiplier
    dens = np.ones(N)
    dens = dens + face * 4.0 + head * 1.5
    dens *= (1 - inner) * (1 - smooth01(0.4, 0.8, lip))
    dens *= smooth01(EYE_R * 1.25, EYE_R * 1.6, eye_d)
    dens *= smooth01(0.028, 0.036, nose_d)

    # ---------------- flow direction (before tangent projection)
    flow = np.tile(np.array([-1.0, 0.0, -0.35]), (N, 1))
    down = np.array([0.0, 0.0, -1.0])
    front = smooth01(0.1, 0.6, fwd) * (1 - head)
    flow = flow * (1 - front[:, None]) + (down + np.array([-0.15, 0, 0]))[None, :] * front[:, None]
    flow = flow * (1 - legs[:, None]) + down[None, :] * legs[:, None]
    side_face = head * smooth01(0.3, 0.8, np.abs(n[:, 1]))
    flow = flow + side_face[:, None] * np.array([0.0, 0.0, -0.5])
    for sy in (1, -1):
        ear = smooth01(1.4, 0.9, ellv(p, (0.412, 0.093 * sy, 0.853), (0.05, 0.03, 0.075)))
        flow = flow * (1 - ear[:, None]) + np.array([-0.05, 0.0, -1.0])[None, :] * ear[:, None]
    flow = np.where(tail[:, None], np.array([-0.3, 0.0, -1.0])[None, :], flow)
    # muzzle hairs point back towards the eyes/cheeks
    flow = flow + (face * 0.4)[:, None] * np.array([-1.0, 0, 0])
    flow += RNG.normal(size=flow.shape) * 0.05

    # lips: hairless black band on the geometry bordering the mouth interior
    near = np.nonzero((x > 0.44) & (z > 0.72) & (z < 0.88) & (ay < 0.08))[0]
    inside = near[inner[near] > 0.5]
    if len(inside):
        kd = KDTree(len(inside))
        for k, vi in enumerate(inside):
            kd.insert(p[vi], k)
        kd.balance()
        band = np.zeros(N)
        for vi in near:
            if inner[vi] <= 0.5:
                band[vi] = kd.find(p[vi])[2]
            else:
                band[vi] = 0.0
        lipb = np.zeros(N)
        lipb[near] = smooth01(0.0075, 0.003, band[near]) * (inner[near] <= 0.5)
        lip = np.maximum(lip, lipb)
    jw = jawness(p)
    mzone = (x > 0.475) & (x < 0.67) & (z > 0.70) & (z < 0.86) & (ay < 0.065) & (inner < 0.5)
    band = np.maximum(column_band(p, mzone & (jw < 0.5), True, 0.0065, 0.004, 0.009),   # upper lip margin
                      column_band(p, mzone & (jw >= 0.5), False, 0.0045, 0.004, 0.009))  # lower lip margin
    band = smooth01(0.3, 0.6, smooth_scalar(edges, band, 6))
    lip = np.maximum(lip, band)
    skin = skin * (1 - lip[:, None]) + np.array([0.012, 0.009, 0.009])[None, :] * lip[:, None]
    wet = np.clip(np.maximum(wet, lip * 0.3), 0, 1)
    dens *= 1 - smooth01(0.3, 0.9, lip)
    col = col * (1 - lip[:, None]) + BLACK[None, :] * lip[:, None]

    ear_all = np.zeros(N)
    for sy in (1, -1):
        ear_all = np.maximum(ear_all, smooth01(1.6, 0.9, ellv(p, (0.405, 0.086 * sy, 0.842), (0.06, 0.035, 0.09))))
    return dict(ear=ear_all, color=np.clip(col, 0, 1), skin=np.clip(skin, 0, 1), length=np.clip(L, 0.002, 0.12),
                density=np.clip(dens, 0, None), flow=flow, wet=wet)


def set_point_attr(me, name, kind, data):
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    if kind == "FLOAT_COLOR":
        a = me.color_attributes.new(name, "FLOAT_COLOR", "POINT")
        rgba = np.concatenate([data, np.ones((len(data), 1))], 1)
        a.data.foreach_set("color", rgba.astype(np.float32).ravel())
    elif kind == "FLOAT":
        a = me.attributes.new(name, "FLOAT", "POINT")
        a.data.foreach_set("value", data.astype(np.float32).ravel())
    elif kind == "FLOAT_VECTOR":
        a = me.attributes.new(name, "FLOAT_VECTOR", "POINT")
        a.data.foreach_set("vector", data.astype(np.float32).ravel())
    return a


def mesh_arrays(me):
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    nor = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("normal", nor)
    return co.reshape(-1, 3).astype(np.float64), nor.reshape(-1, 3).astype(np.float64)


# --------------------------------------------------------------------------- accessory meshes
def mesh_from_pydata(name, verts, faces, coll, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    for p in me.polygons:
        p.use_smooth = smooth
    return link(bpy.data.objects.new(name, me), coll)


def uv_sphere(name, coll, radius=1.0, seg=32, rings=16):
    verts, faces = [], []
    for i in range(rings + 1):
        th = math.pi * i / rings
        for j in range(seg):
            ph = 2 * math.pi * j / seg
            verts.append((radius * math.cos(th), radius * math.sin(th) * math.cos(ph),
                          radius * math.sin(th) * math.sin(ph)))
    for i in range(rings):
        for j in range(seg):
            a = i * seg + j
            b = i * seg + (j + 1) % seg
            faces.append((a, b, b + seg, a + seg))
    obj = mesh_from_pydata(name, verts, faces, coll)
    # weld poles
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bm.to_mesh(obj.data)
    bm.free()
    return obj


def build_eyes(coll):
    eyes = []
    for c, side in zip(EYES, ("L", "R")):
        o = uv_sphere(f"Dog_Eye.{side}", coll, 1.0, 32, 16)  # +X is the gaze axis
        o.location = c
        o.scale = (EYE_R * 0.9, EYE_R, EYE_R * 0.92)
        sy = 1 if side == "L" else -1
        gaze = Vector((1.0, 0.55 * sy, 0.1)).normalized()
        o.rotation_mode = "QUATERNION"
        o.rotation_quaternion = Vector((1, 0, 0)).rotation_difference(gaze)
        eyes.append(o)
    return eyes


def build_nose(coll):
    """Leathery nose pad sculpted from a sphere: flat front, comma nostrils, philtrum groove."""
    import bmesh
    me = bpy.data.meshes.new("Dog_Nose")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=40, radius=1.0)
    nost = [Vector((0.78, 0.50 * sy, -0.22)).normalized() for sy in (1, -1)]
    for v in bm.verts:
        u = v.co.normalized()
        x, y, z = v.co
        push = 0.0
        for k, c in enumerate(nost):
            sy = 1 if k == 0 else -1
            ang = u.angle(c)
            # comma: round opening that trails backwards/outwards along the side
            tail = max(0.0, (abs(y) - abs(c.y))) * (1 if y * sy > 0 else 0)
            push = max(push, 0.55 * math.exp(-(ang / 0.20) ** 2) + 0.25 * math.exp(-(ang / 0.34) ** 2) * min(tail * 3, 1))
        # philtrum groove running down the front
        groove = 0.28 * math.exp(-(y / 0.10) ** 2) * max(0.0, -z) * max(0.0, x)
        push = max(push, groove)
        r = 1.0 - push
        x, y, z = u.x * r, u.y * r, u.z * r
        # flatten the top & front into a pad
        if z > 0.35:
            z = 0.35 + (z - 0.35) * 0.55
        if x > 0.45:
            x = 0.45 + (x - 0.45) * 0.5
        v.co = Vector((x * 0.024, y * (0.026 + 0.004 * z), z * 0.021))
    bm.to_mesh(me)
    bm.free()
    o = link(bpy.data.objects.new("Dog_Nose", me), coll)
    o.location = (NOSE_TIP[0] - 0.013, 0, NOSE_TIP[2] + 0.002)
    o.rotation_euler = (0, R(-10), 0)
    sub = o.modifiers.new("Subdiv", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 2
    for p in me.polygons:
        p.use_smooth = True
    return o


def build_tongue(coll):
    """Lofted flat tongue with a medial groove hanging out of the mouth."""
    pts = [Vector(p) for p in TONGUE_SPINE]
    # Catmull-Rom spine
    spine = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, len(pts) - 1)]
        for k in range(8):
            t = k / 8
            t2, t3 = t * t, t * t * t
            spine.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                                (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    spine.append(pts[-1])
    ns = len(spine)
    ring = 24
    verts, faces = [], []
    for i, s in enumerate(spine):
        u = i / (ns - 1)
        t = (spine[min(i + 1, ns - 1)] - spine[max(i - 1, 0)]).normalized()
        side = Vector((0, 1, 0))
        up = t.cross(side).normalized() * -1  # dorsal
        side = up.cross(t).normalized()
        w = 0.018 + 0.008 * math.sin(min(u * 1.3, 1.0) * math.pi * 0.5)
        w *= 1.0 - 0.55 * max(0, (u - 0.85) / 0.15) ** 2
        th = 0.0065 * (1 - 0.4 * u)
        for j in range(ring):
            a = 2 * math.pi * j / ring
            cy, cz = math.cos(a), math.sin(a)
            groove = -0.0025 * math.exp(-(cy / 0.25) ** 2) if cz > 0 else 0.0
            off = side * (w * cy) + up * (th * cz * (1.0 if cz < 0 else 0.8) + groove * cz)
            verts.append(tuple(s + off))
    for i in range(ns - 1):
        for j in range(ring):
            a = i * ring + j
            b = i * ring + (j + 1) % ring
            faces.append((a, b, b + ring, a + ring))
    tip = len(verts)
    verts.append(tuple(spine[-1] + (spine[-1] - spine[-2]).normalized() * 0.003))
    for j in range(ring):
        a = (ns - 1) * ring + j
        b = (ns - 1) * ring + (j + 1) % ring
        faces.append((a, b, tip))
    o = mesh_from_pydata("Dog_Tongue", verts, faces, coll)
    sub = o.modifiers.new("Subdiv", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 2
    return o


def cone(name, coll, base, tip, r):
    base, tip = Vector(base), Vector(tip)
    d = (tip - base)
    seg = 10
    q = Vector((0, 0, 1)).rotation_difference(d.normalized())
    verts = [tuple(base + q @ Vector((r * math.cos(2 * math.pi * j / seg), r * math.sin(2 * math.pi * j / seg), 0)))
             for j in range(seg)]
    verts.append(tuple(base + q @ Vector((0, 0, d.length * 0.55))))
    verts.append(tuple(tip))
    faces = [(j, (j + 1) % seg, seg) for j in range(seg)]
    return verts, faces


def build_teeth(coll):
    objs = {}
    for part, spec in (("Upper", [((0.585, 0.024, 0.800), (0.588, 0.023, 0.783), 0.0038),
                                  ((0.605, 0.012, 0.802), (0.606, 0.011, 0.794), 0.0022),
                                  ((0.608, 0.004, 0.803), (0.609, 0.004, 0.795), 0.002)]),
                       ("Lower", [(jaw((0.585, 0.020, 0.804)), jaw((0.583, 0.019, 0.821)), 0.0034),
                                  (jaw((0.600, 0.009, 0.806)), jaw((0.600, 0.009, 0.814)), 0.0018)])):
        verts, faces = [], []
        for base, tip, r in spec:
            for sy in (1, -1):
                b = (base[0], base[1] * sy, base[2])
                t = (tip[0], tip[1] * sy, tip[2])
                v, f = cone("", coll, b, t, r)
                off = len(verts)
                verts += v
                faces += [tuple(i + off for i in fc) for fc in f]
        objs[part] = mesh_from_pydata(f"Dog_Teeth_{part}", verts, faces, coll)
    return objs


def build_whiskers(coll):
    cu = bpy.data.hair_curves.new("Dog_Whiskers")
    roots = []
    for sy in (1, -1):
        for i in range(9):
            roots.append((0.575 + 0.012 * (i % 3) + RNG.normal() * 0.002, (0.034 - 0.002 * (i // 3)) * sy,
                          0.835 - 0.006 * (i // 3), sy))
    n = len(roots)
    k = 5
    cu.add_curves([k] * n)
    pos = []
    for rx, ry, rz, sy in roots:
        d = Vector((-0.35 + RNG.normal() * 0.15, 1.0 * sy, -0.3 + RNG.normal() * 0.15)).normalized()
        L = RNG.uniform(0.035, 0.06)
        p = Vector((rx, ry, rz))
        for j in range(k):
            t = j / (k - 1)
            pos.append(tuple(p + d * L * t + Vector((0, 0, -0.012 * t * t))))
    cu.attributes["position"].data.foreach_set("vector", np.array(pos, np.float32).ravel())
    rad = cu.attributes.new("radius", "FLOAT", "POINT")
    rad.data.foreach_set("value", np.tile(np.linspace(0.00022, 0.00004, k), n).astype(np.float32))
    return link(bpy.data.objects.new("Dog_Whiskers", cu), coll)


# --------------------------------------------------------------------------- materials
def nodes_mat(name):
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return mat, nt, out


def principled(nt, **kw):
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    for k, v in kw.items():
        b.inputs[k].default_value = v
    return b


def mat_skin():
    mat, nt, out = nodes_mat("Dog_Skin")
    col = nt.nodes.new("ShaderNodeAttribute")
    col.attribute_name = "skin"
    wet = nt.nodes.new("ShaderNodeAttribute")
    wet.attribute_name = "wet"
    b = principled(nt, **{"Roughness": 0.75, "Subsurface Weight": 0.05})
    b.inputs["Subsurface Radius"].default_value = (0.8, 0.25, 0.15)
    b.inputs["Subsurface Scale"].default_value = 0.004
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.8
    rough.inputs["To Max"].default_value = 0.22
    nt.links.new(wet.outputs["Fac"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], b.inputs["Roughness"])
    nt.links.new(col.outputs["Color"], b.inputs["Base Color"])
    coat = nt.nodes.new("ShaderNodeMath")
    coat.operation = "MULTIPLY"
    coat.inputs[1].default_value = 0.6
    nt.links.new(wet.outputs["Fac"], coat.inputs[0])
    nt.links.new(coat.outputs[0], b.inputs["Coat Weight"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return mat


def set_in(node, name, val):
    if name in node.inputs:
        node.inputs[name].default_value = val


def mat_fur():
    mat, nt, out = nodes_mat("Dog_Fur")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "fur_color"
    h = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    variant = os.environ.get("FUR_VARIANT", "huang")
    if variant == "huang":
        h.model = "HUANG"
    h.parametrization = "COLOR"
    h.inputs["Roughness"].default_value = 0.27
    if "Radial Roughness" in h.inputs and h.inputs["Radial Roughness"].enabled:
        h.inputs["Radial Roughness"].default_value = 0.5
    set_in(h, "Coat", 0.0)
    set_in(h, "IOR", 1.5)
    set_in(h, "Random Roughness", 0.15)
    if "Random Color" in h.inputs:
        h.inputs["Random Color"].default_value = 0.08
    if variant == "huang":
        h.inputs["Reflection"].default_value = float(os.environ.get("FUR_R", "0.3"))
        h.inputs["Secondary Reflection"].default_value = 0.6
    elif variant == "lowior":
        h.inputs["IOR"].default_value = 1.3
    # darken tips slightly less than roots -> use Hair Info intercept to brighten tips
    info = nt.nodes.new("ShaderNodeHairInfo")
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 0.2
    mr.inputs["To Max"].default_value = 0.0
    nt.links.new(info.outputs["Intercept"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], mix.inputs["Factor"])
    nt.links.new(attr.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = (0.8, 0.72, 0.62, 1)
    nt.links.new(mix.outputs[2], h.inputs["Color"])
    nt.links.new(h.outputs[0], out.inputs["Surface"])
    return mat


def mat_whisker():
    mat, nt, out = nodes_mat("Dog_Whisker")
    h = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    h.parametrization = "COLOR"
    h.inputs["Color"].default_value = (0.6, 0.58, 0.55, 1)
    h.inputs["Roughness"].default_value = 0.2
    nt.links.new(h.outputs[0], out.inputs["Surface"])
    return mat


def mat_eye():
    mat, nt, out = nodes_mat("Dog_Eye")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    # radial distance from gaze axis (+X)
    y2 = nt.nodes.new("ShaderNodeVectorMath")
    y2.operation = "LENGTH"
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["Y"], comb.inputs["Y"])
    nt.links.new(sep.outputs["Z"], comb.inputs["Z"])
    nt.links.new(comb.outputs[0], y2.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.30
    cr.elements[0].color = (0.004, 0.003, 0.002, 1)
    cr.elements[1].position = 0.36
    cr.elements[1].color = (0.035, 0.016, 0.006, 1)
    e = cr.elements.new(0.62)
    e.color = (0.02, 0.01, 0.005, 1)
    e = cr.elements.new(0.72)
    e.color = (0.02, 0.012, 0.008, 1)
    nt.links.new(y2.outputs["Value"], ramp.inputs[0])
    b = principled(nt, **{"Roughness": 0.35, "Coat Weight": 1.0, "Coat Roughness": 0.02, "Coat IOR": 1.376})
    nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return mat


def mat_nose():
    mat, nt, out = nodes_mat("Dog_Nose")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 900.0
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(vor.outputs["Distance"], bump.inputs["Height"])
    b = principled(nt, **{"Base Color": (0.003, 0.0027, 0.0027, 1), "Roughness": 0.55,
                          "Specular IOR Level": 0.12})
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return mat


def mat_tongue():
    mat, nt, out = nodes_mat("Dog_Tongue")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noi = nt.nodes.new("ShaderNodeTexNoise")
    noi.inputs["Scale"].default_value = 700.0
    nt.links.new(tc.outputs["Object"], noi.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.0002
    nt.links.new(noi.outputs["Fac"], bump.inputs["Height"])
    b = principled(nt, **{"Base Color": (0.30, 0.045, 0.065, 1), "Roughness": 0.4, "Subsurface Weight": 0.15,
                          "Subsurface Scale": 0.002, "Coat Weight": 0.15, "Coat Roughness": 0.25})
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.25)
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return mat


def mat_teeth():
    mat, nt, out = nodes_mat("Dog_Teeth")
    b = principled(nt, **{"Base Color": (0.80, 0.74, 0.60, 1), "Roughness": 0.25, "Subsurface Weight": 0.2,
                          "Subsurface Scale": 0.002})
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return mat


# --------------------------------------------------------------------------- rig
BONES = [
    # name, head, tail, parent, deform
    ("root", (0, 0, 0), (0.25, 0, 0), None, False),
    ("spine_01", (-0.42, 0, 0.58), (-0.16, 0, 0.585), "root", True),
    ("spine_02", (-0.16, 0, 0.585), (0.06, 0, 0.59), "spine_01", True),
    ("spine_03", (0.06, 0, 0.59), (0.25, 0, 0.60), "spine_02", True),
    ("neck_01", (0.25, 0, 0.60), (0.33, 0, 0.71), "spine_03", True),
    ("neck_02", (0.33, 0, 0.71), (0.40, 0, 0.82), "neck_01", True),
    ("head", (0.40, 0, 0.82), (0.63, 0, 0.855), "neck_02", True),
    ("jaw", JAW_PIVOT, jaw((0.61, 0, 0.80)), "head", True),
    ("tongue_01", TONGUE_SPINE[0], TONGUE_SPINE[2], "jaw", True),
    ("tongue_02", TONGUE_SPINE[2], TONGUE_SPINE[4], "tongue_01", True),
    ("tongue_03", TONGUE_SPINE[4], TONGUE_SPINE[7], "tongue_02", True),
    ("ear.L", (0.405, 0.082, 0.905), (0.418, 0.105, 0.79), "head", True),
    ("ear.R", (0.405, -0.082, 0.905), (0.418, -0.105, 0.79), "head", True),
    ("tail_01", (-0.46, 0, 0.61), (-0.535, 0, 0.51), "spine_01", True),
    ("tail_02", (-0.535, 0, 0.51), (-0.555, 0, 0.37), "tail_01", True),
    ("tail_03", (-0.555, 0, 0.37), (-0.53, 0, 0.21), "tail_02", True),
]
for s, sy in (("L", 1), ("R", -1)):
    y, yh = 0.10 * sy, 0.095 * sy
    BONES += [
        (f"shoulder.{s}", (0.17, 0.06 * sy, 0.64), (0.22, y, 0.49), "spine_03", True),
        (f"upper_arm.{s}", (0.22, y, 0.49), (0.185, y, 0.31), f"shoulder.{s}", True),
        (f"forearm.{s}", (0.185, y, 0.31), (0.195, y, 0.07), f"upper_arm.{s}", True),
        (f"front_paw.{s}", (0.195, y, 0.07), (0.26, y, 0.02), f"forearm.{s}", True),
        (f"thigh.{s}", (-0.36, yh, 0.56), (-0.31, yh, 0.34), "spine_01", True),
        (f"shin.{s}", (-0.31, yh, 0.34), (-0.43, yh, 0.16), f"thigh.{s}", True),
        (f"hind_foot.{s}", (-0.43, yh, 0.16), (-0.41, yh, 0.06), f"shin.{s}", True),
        (f"hind_paw.{s}", (-0.41, yh, 0.06), (-0.34, yh, 0.02), f"hind_foot.{s}", True),
        (f"front_ik.{s}", (0.195, y, 0.07), (0.195, y, 0.0), "root", False),
        (f"hind_ik.{s}", (-0.41, yh, 0.06), (-0.41, yh, 0.0), "root", False),
    ]


def build_rig(coll):
    arm = bpy.data.armatures.new("Dog_Rig")
    arm.display_type = "OCTAHEDRAL"
    rig = link(bpy.data.objects.new("Dog_Rig", arm), coll)
    rig.show_in_front = True
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    for name, h, t, parent, deform in BONES:
        b = arm.edit_bones.new(name)
        b.head, b.tail = h, t
        b.use_deform = deform
        if parent:
            b.parent = eb[parent]
            b.use_connect = (Vector(eb[parent].tail) - Vector(h)).length < 1e-5 and parent != "root"
        eb[name] = b
    # roll bones consistently (Z up / lateral)
    for b in arm.edit_bones:
        b.align_roll(Vector((0, 0, 1)) if abs(b.vector.normalized().z) < 0.8 else Vector((1, 0, 0)))
    bpy.ops.object.mode_set(mode="OBJECT")
    # IK on legs
    for s in ("L", "R"):
        for chain, tgt in ((f"forearm.{s}", f"front_ik.{s}"), (f"shin.{s}", f"hind_ik.{s}")):
            c = rig.pose.bones[chain].constraints.new("IK")
            c.target = rig
            c.subtarget = tgt
            c.chain_count = 2
            c.use_tail = True
    # bone collections for tidiness
    try:
        deform = arm.collections.new("Deform")
        ctrl = arm.collections.new("Controls")
        for b in arm.bones:
            (deform if b.use_deform else ctrl).assign(b)
    except Exception:
        pass
    return rig


def seg_dist(p, a, b):
    a, b = np.array(a), np.array(b)
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def skin_body(body, rig, coat):
    """Bone-heat automatic weights, then split the lower jaw cleanly from the skull."""
    arm = rig.data
    tongue_bones = [b for b in arm.bones if b.name.startswith("tongue")]
    for b in tongue_bones:
        b.use_deform = False
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    for b in tongue_bones:
        b.use_deform = True
    me = body.data
    co, _ = mesh_arrays(me)
    nv = len(co)
    names = [g.name for g in body.vertex_groups]
    W = np.zeros((nv, len(names)))
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    unweighted = W.sum(1) < 1e-4
    log(f"auto weights: {int(unweighted.sum())} unweighted verts")
    if unweighted.any():
        # fall back to nearest deform bone for anything heat weighting missed
        deform = [b for b in BONES if b[4] and not b[0].startswith("tongue")]
        D = np.stack([seg_dist(co[unweighted], b[1], b[2]) for b in deform], 1)
        nearest = np.argmin(D, 1)
        for k, vi in enumerate(np.nonzero(unweighted)[0]):
            W[vi, names.index(deform[nearest[k]][0])] = 1.0
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    hi, ji = names.index("head"), names.index("jaw")
    jf = jawness(co) * smooth01(0.43, 0.47, x)
    inface = (x > 0.40) & (z > 0.70)
    tot = W[:, hi] + W[:, ji]
    W[inface, ji] = (tot * jf)[inface]
    W[inface, hi] = (tot * (1 - jf))[inface]
    W /= W.sum(1, keepdims=True) + 1e-12
    for i, n in enumerate(names):
        vg = body.vertex_groups[n]
        vg.remove(list(range(nv)))
        nz = np.nonzero(W[:, i] > 0.001)[0]
        for vi in nz:
            vg.add([int(vi)], float(W[vi, i]), "REPLACE")
    mod = body.modifiers["Armature"]
    mod.use_deform_preserve_volume = True


def skin_tongue(tongue, rig):
    co, _ = mesh_arrays(tongue.data)
    tb = [b for b in BONES if b[0].startswith("tongue")]
    D = np.stack([seg_dist(co, b[1], b[2]) for b in tb], 1)
    w = 1.0 / (D ** 3 + 1e-9)
    w /= w.sum(1, keepdims=True)
    for i, b in enumerate(tb):
        vg = tongue.vertex_groups.new(name=b[0])
        for vi in np.nonzero(w[:, i] > 0.01)[0]:
            vg.add([int(vi)], float(w[vi, i]), "REPLACE")
    m = tongue.modifiers.new("Armature", "ARMATURE")
    m.object = rig
    tongue.modifiers.move(len(tongue.modifiers) - 1, 0)
    tongue.parent = rig


def parent_to_bone(obj, rig, bone):
    mw = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone
    bpy.context.view_layer.update()
    obj.matrix_world = mw


# --------------------------------------------------------------------------- fur
def build_fur(body, maps, coll, n_strands):
    t0 = time.time()
    me = body.data
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tri_v = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tri_v)
    tri_v = tri_v.reshape(-1, 3)
    tri_l = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tri_l)
    tri_l = tri_l.reshape(-1, 3)
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers["UVMap"].data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    co, nor = mesh_arrays(me)

    P = co[tri_v]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    dens = maps["density"][tri_v].mean(1)
    prob = area * dens
    prob /= prob.sum()
    ti = RNG.choice(nt, size=n_strands, p=prob)
    r1, r2 = RNG.random(n_strands), RNG.random(n_strands)
    s1 = np.sqrt(r1)
    bc = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)

    def interp(a):
        return (a[tri_v[ti]] * bc[..., None]).sum(1) if a.ndim == 2 else (a[tri_v[ti]] * bc).sum(1)

    root = interp(co)
    n = interp(nor)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    col = interp(maps["color"])
    # mixed hairs along pattern boundaries: each strand picks one side of the transition
    lum = col.mean(1)
    edge = (lum > 0.02) & (lum < 0.6)
    pick = RNG.random(n_strands)
    tw = np.clip((lum - 0.006) / 0.75, 0, 1)
    col[edge & (pick > tw) & (RNG.random(n_strands) < 0.6)] *= 0.25
    col *= RNG.uniform(0.85, 1.12, (n_strands, 1))
    L = interp(maps["length"]) * RNG.uniform(0.75, 1.15, n_strands)
    flow = interp(maps["flow"])
    earg = interp(maps["ear"])
    suv = (uv[tri_l[ti]] * bc[..., None]).sum(1)

    def tangent(v, nn):
        v = v - (v * nn).sum(1, keepdims=True) * nn
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    ft = tangent(flow, n)
    grav = np.array([0.0, 0.0, -1.0])
    long_ = smooth01(0.015, 0.06, L)  # 0 short, 1 long fur
    lift = 0.30 - 0.12 * long_
    # random perpendicular for waves
    side = np.cross(n, ft)
    wave_ph = RNG.uniform(0, 6.28, n_strands)
    wave_amp = (0.04 + 0.14 * long_) * L
    K = 8
    pts = np.zeros((n_strands, K, 3))
    pts[:, 0] = root
    d = ft + n * lift[:, None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    seg = L / (K - 1)
    jitter = RNG.normal(size=(n_strands, 3)) * 0.12
    for k in range(1, K):
        t = k / (K - 1)
        gt = tangent(np.tile(grav, (n_strands, 1)), n)
        target = ft * (1 - 0.5 * long_[:, None] * t) + gt * (0.5 * long_[:, None] * t) + n * (lift * (1 - t) ** 2)[:, None] * 0.6
        target += jitter
        d = d * 0.6 + target * 0.4
        d += grav * (0.55 * long_ * t + 0.08 * t + 0.9 * earg * t)[:, None]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts[:, k] = pts[:, k - 1] + d * seg[:, None]
    # waves
    tt = np.linspace(0, 1, K)
    wav = np.sin(tt[None, :] * 2.6 * math.pi + wave_ph[:, None]) * (tt[None, :] ** 1.2)
    pts += side[:, None, :] * (wav * wave_amp[:, None])[..., None]

    # clumping towards nearby guide strands (coarser guides -> bigger locks for long fur)
    near = np.empty(n_strands, np.int64)
    ndist = np.empty(n_strands)
    is_long = L > 0.03
    for sel, G in ((~is_long, max(2000, n_strands // 14)), (is_long, max(500, n_strands // 90))):
        ids = np.nonzero(sel)[0]
        if not len(ids):
            continue
        gi = RNG.choice(ids, min(G, len(ids)), replace=False)
        kd = KDTree(len(gi))
        for i, g in enumerate(gi):
            kd.insert(root[g], i)
        kd.balance()
        for i in ids:
            _, j, dd = kd.find(root[i])
            near[i] = gi[j]
            ndist[i] = dd
    g_pts = pts[near]
    g_root = root[near]
    scale = (L / (L[near] + 1e-9))[:, None, None]
    clump = (0.25 + 0.6 * long_) * smooth01(0.03, 0.012, ndist)
    target = root[:, None, :] + (g_pts - g_root[:, None, :]) * scale
    cw = clump[:, None] * (tt[None, :] ** 1.3)
    # pull toward guide's path, converging tips to the guide
    conv = target + (g_pts - target) * (cw * 0.7)[..., None]
    pts = pts + (conv - pts) * cw[..., None]

    # split short and long strands to use fewer points for short fur
    short = L < 0.012
    order = np.concatenate([np.nonzero(short)[0], np.nonzero(~short)[0]])
    ns = int(short.sum())
    sizes = [4] * ns + [K] * (n_strands - ns)
    pos = np.concatenate([pts[order[:ns]][:, [0, 2, 5, 7]].reshape(-1, 3), pts[order[ns:]].reshape(-1, 3)])
    root_r = np.where(L < 0.012, 0.00018, 0.00032)
    rad_s = (root_r[order[:ns], None] * np.array([1.0, 0.7, 0.35, 0.1])[None]).ravel()
    rad_l = (root_r[order[ns:], None] * np.linspace(1.0, 0.12, K)[None]).ravel()

    cu = bpy.data.hair_curves.new("Dog_Fur")
    cu.add_curves(sizes)
    cu.attributes["position"].data.foreach_set("vector", pos.astype(np.float32).ravel())
    ra = cu.attributes.new("radius", "FLOAT", "POINT")
    ra.data.foreach_set("value", np.concatenate([rad_s, rad_l]).astype(np.float32))
    ca = cu.attributes.new("fur_color", "FLOAT_COLOR", "CURVE")
    c4 = np.concatenate([col[order], np.ones((n_strands, 1))], 1)
    ca.data.foreach_set("color", c4.astype(np.float32).ravel())
    su = cu.attributes.new("surface_uv_coordinate", "FLOAT2", "CURVE")
    su.data.foreach_set("vector", suv[order].astype(np.float32).ravel())
    cu.surface = body
    cu.surface_uv_map = "UVMap"
    obj = link(bpy.data.objects.new("Dog_Fur", cu), coll)
    log(f"fur: {n_strands} strands ({ns} short) in {time.time() - t0:.1f}s")
    return obj


def reference_albedo(path):
    """Classify the reference photo into fur albedo (black / tan / white-grey), blurred to ignore
    individual hair highlights. Returns (albedo HxWx3, valid HxW) with rows top-first."""
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.array(img.pixels[:], np.float32).reshape(h, w, 4)[::-1, :, :3].astype(np.float64)
    k = 4
    pad = np.pad(px, ((k, k), (k, k), (0, 0)), mode="edge")
    blur = np.zeros_like(px)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            blur += pad[k + dy:k + dy + h, k + dx:k + dx + w]
    blur /= (2 * k + 1) ** 2
    r, g, b = blur[..., 0], blur[..., 1], blur[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    bg = (b - r > 0.22) & (g > 0.45) & (b > 0.6)
    pink = (r - g > 0.18) & (b - g > 0.02)
    t = smooth01(0.07, 0.17, r - b) * smooth01(0.10, 0.2, lum)          # rust / tan
    wv = smooth01(0.26, 0.50, lum) * smooth01(0.05, 0.0, b - r - 0.4 * smooth01(0.5, 0.8, lum)) * (1 - t)  # warm/neutral light = white fur; cool = black-fur sheen
    alb = (BLACK[None, None] * (1 - t - wv)[..., None] + TAN[None, None] * t[..., None]
           + WHITE[None, None] * wv[..., None])
    bpy.data.images.remove(img)
    return alb, ~(bg | pink)


def project_reference_markings(fur, body, cam_obj, ref_path):
    """Recolour the fur strands the camera sees so their markings follow the reference photo."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils.bvhtree import BVHTree
    t0 = time.time()
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    alb, valid = reference_albedo(ref_path)
    H, W = valid.shape
    ev = fur.evaluated_get(dg)
    npts = len(ev.data.points)
    pos = np.empty(npts * 3, np.float32)
    ev.data.attributes["position"].data.foreach_get("vector", pos)
    mw = np.array(fur.matrix_world)
    pos = pos.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
    ncurves = len(fur.data.curves)
    offs = np.empty(ncurves + 1, np.int32)
    fur.data.curve_offset_data.foreach_get("value", offs)
    size = np.diff(offs)
    root = pos[offs[:-1]]
    sample = pos[offs[:-1] + np.maximum(1, (size * 0.45).astype(int))]
    # project with the camera's exact framing
    mv = np.array(cam_obj.matrix_world.inverted())
    cd = cam_obj.data
    kf = cd.lens / cd.sensor_width

    def proj(p):
        pc = p @ mv[:3, :3].T + mv[:3, 3]
        z = np.maximum(-pc[:, 2], 1e-6)
        x = W * (0.5 + kf * pc[:, 0] / z - cd.shift_x)
        y = H * 0.5 + W * (kf * pc[:, 1] / z - cd.shift_y)
        return x, H - y

    sx, sy = proj(sample)
    xi, yi = np.round(sx).astype(int), np.round(sy).astype(int)
    inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    cand = np.nonzero(inside)[0]
    cand = cand[valid[yi[cand], xi[cand]]]
    # the face keeps its sculpted procedural markings (sub-cm misalignment there reads as blotches)
    rest = np.empty(npts * 3, np.float32)
    fur.data.attributes["position"].data.foreach_get("vector", rest)
    rroot = rest.reshape(-1, 3)[offs[:-1]]
    face = (rroot[:, 0] > 0.44) & (rroot[:, 2] > 0.72) & (np.abs(rroot[:, 1]) < 0.075)
    if os.environ.get("DOG_PROJECT_FACE", "1") == "1":
        face[:] = False
    cand = cand[~face[cand]]
    # visibility: is the root the first body surface hit from the camera?
    bvh = BVHTree.FromObject(body, dg)
    cam = cam_obj.matrix_world.translation
    vis = np.zeros(ncurves, bool)
    for c in cand:
        rp = Vector(sample[c])  # the part of the strand that shows, not its (often hidden) root
        d = rp - cam
        dist = d.length
        hit = bvh.ray_cast(cam, d / dist, dist - 0.004)
        vis[c] = hit[0] is None
    sel = np.nonzero(vis)[0]
    col = np.empty(ncurves * 4, np.float32)
    fur.data.attributes["fur_color"].data.foreach_get("color", col)
    col = col.reshape(-1, 4)
    new = alb[yi[sel], xi[sel]]
    # smooth over neighbouring strands on the surface (removes speckle from sub-cm misregistration)
    kd = KDTree(len(sel))
    for k_, c in enumerate(sel):
        kd.insert(rroot[c], k_)
    kd.balance()
    sm = np.empty_like(new)
    for k_, c in enumerate(sel):
        idx = [i for _, i, _ in kd.find_range(rroot[c], 0.005)]
        sm[k_] = new[idx].mean(0)
    new = sm
    # grey = a mix of black and white locks: dither neutral colours with a coherent lock-scale noise
    rr = rroot[sel].astype(np.float64)
    lum = new @ np.array([0.2126, 0.7152, 0.0722])
    lb, lw = BLACK @ np.array([0.2126, 0.7152, 0.0722]), WHITE @ np.array([0.2126, 0.7152, 0.0722])
    wfrac = np.clip((lum - lb) / (lw - lb), 0, 1)
    neutral = smooth01(0.06, 0.02, np.abs(new[:, 0] - new[:, 2]) / (lum + 0.02) * 0.1)
    thr = np.clip(0.5 + 0.7 * fbm_noise(rr, 110.0, 11, octaves=3), 0.02, 0.98)
    crisp = BLACK[None] + (WHITE - BLACK)[None] * smooth01(thr - 0.06, thr + 0.06, wfrac ** 1.3)[:, None]
    new = new * (1 - neutral[:, None]) + crisp * neutral[:, None]
    # keep a little per-strand variation
    new = new * RNG.uniform(0.88, 1.1, (len(sel), 1))
    col[sel, :3] = new
    fur.data.attributes["fur_color"].data.foreach_set("color", col.ravel())
    fur.data.update_tag()
    # the skin under the coat shows through between strands: give it the same markings
    ev_b = body.evaluated_get(dg)
    me_b = ev_b.to_mesh()
    nv = len(me_b.vertices)
    bco = np.empty(nv * 3, np.float32)
    me_b.vertices.foreach_get("co", bco)
    ev_b.to_mesh_clear()
    mwb = np.array(body.matrix_world)
    bco = bco.reshape(-1, 3).astype(np.float64) @ mwb[:3, :3].T + mwb[:3, 3]
    bx, by = proj(bco)
    bxi, byi = np.round(bx).astype(int), np.round(by).astype(int)
    ok = np.nonzero((bxi >= 0) & (bxi < W) & (byi >= 0) & (byi < H))[0]
    ok = ok[valid[byi[ok], bxi[ok]]]
    rest_b = np.empty(nv * 3, np.float32)
    body.data.vertices.foreach_get("co", rest_b)
    rest_b = rest_b.reshape(-1, 3)
    ok = ok[~((rest_b[ok, 0] > 0.44) & (rest_b[ok, 2] > 0.72) & (np.abs(rest_b[ok, 1]) < 0.075))]  # not the face
    seen = []
    for v in ok:
        d = Vector(bco[v]) - cam
        L = d.length
        if bvh.ray_cast(cam, d / L, L - 0.003)[0] is None:
            seen.append(v)
    seen = np.array(seen, int)
    skin = np.empty(nv * 4, np.float32)
    body.data.color_attributes["skin"].data.foreach_get("color", skin)
    skin = skin.reshape(-1, 4)
    skin[seen, :3] = alb[byi[seen], bxi[seen]] * 0.8
    body.data.color_attributes["skin"].data.foreach_set("color", skin.ravel())
    body.data.update()
    log(f"reference markings: {len(sel)} of {ncurves} strands recoloured ({len(cand)} in view) "
        f"in {time.time() - t0:.1f}s")


def fur_deform_modifier(fur):
    ng = bpy.data.node_groups.new("FurFollowSurface", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = ng.nodes.new("NodeGroupInput")
    go = ng.nodes.new("NodeGroupOutput")
    d = ng.nodes.new("GeometryNodeDeformCurvesOnSurface")
    ng.links.new(gi.outputs[0], d.inputs[0])
    ng.links.new(d.outputs[0], go.inputs[0])
    m = fur.modifiers.new("FollowSurface", "NODES")
    m.node_group = ng


# --------------------------------------------------------------------------- world / camera / lights
def srgb_to_lin(c):
    c = np.array(c)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def build_world():
    w = bpy.data.worlds.new("Studio")
    bpy.context.scene.world = w
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Window"], sep.inputs[0])
    # backdrop gradient in screen space (sampled from the reference)
    ramp_x = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp_x.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (*srgb_to_lin((0.36, 0.76, 0.89)), 1)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (*srgb_to_lin((0.39, 0.66, 0.78)), 1)
    e = cr.elements.new(0.55)
    e.color = (*srgb_to_lin((0.39, 0.71, 0.85)), 1)
    nt.links.new(sep.outputs["X"], ramp_x.inputs[0])
    # brighten towards bottom
    ramp_y = nt.nodes.new("ShaderNodeMapRange")
    ramp_y.inputs["From Min"].default_value = 0.0
    ramp_y.inputs["From Max"].default_value = 1.0
    ramp_y.inputs["To Min"].default_value = 1.25
    ramp_y.inputs["To Max"].default_value = 0.88
    nt.links.new(sep.outputs["Y"], ramp_y.inputs["Value"])
    mulc = nt.nodes.new("ShaderNodeVectorMath")
    mulc.operation = "SCALE"
    nt.links.new(ramp_x.outputs[0], mulc.inputs[0])
    nt.links.new(ramp_y.outputs[0], mulc.inputs["Scale"])
    bg_cam = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(mulc.outputs[0], bg_cam.inputs["Color"])
    bg_cam.inputs["Strength"].default_value = 1.0
    bg_light = nt.nodes.new("ShaderNodeBackground")
    bg_light.inputs["Color"].default_value = (*srgb_to_lin((0.55, 0.75, 0.85)), 1)
    bg_light.inputs["Strength"].default_value = 0.08
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(bg_light.outputs[0], mix.inputs[1])
    nt.links.new(bg_cam.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])


def area_light(name, coll, loc, target, size, energy, color=(1, 1, 1)):
    l = bpy.data.lights.new(name, "AREA")
    l.shape = "RECTANGLE"
    l.size, l.size_y = size
    l.energy = energy
    l.color = color
    o = link(bpy.data.objects.new(name, l), coll)
    o.location = loc
    look_at(o, target)
    return o


def look_at(obj, target, roll=0.0):
    d = Vector(target) - Vector(obj.location)
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


KEY_LIGHT = [tuple(float(v) for v in os.environ.get("KEY_LOC", "2.2,-1.0,1.7").split(",")), (0.52, 0, 0.84),
             (1.6, 1.2), float(os.environ.get("KEY_W", "210")), (1.0, 0.97, 0.93)]


def build_lights(coll):
    head = (0.52, 0, 0.84)
    key = area_light("Key", coll, *KEY_LIGHT)
    key.data.spread = R(float(os.environ.get("KEY_SPREAD", "180")))
    area_light("Fill", coll, (0.2, -2.6, 0.9), head, (2.0, 1.4), float(os.environ.get("FILL_W", "8")), (0.85, 0.93, 1.0))
    area_light("Rim", coll, (-0.6, 1.8, 1.6), head, (1.0, 1.0), 90, (0.8, 0.92, 1.0))
    area_light("Chest", coll, (1.5, -1.0, 0.35), (0.3, 0, 0.5), (1.0, 0.6), float(os.environ.get("CHEST_W", "0")), (1.0, 0.98, 0.95))
    area_light("RimR", coll, (1.8, 1.4, 1.2), head, (0.8, 0.8), 140, (0.85, 0.95, 1.0))


CAMS = {
    # name: (location, target, lens, shift_x, shift_y)
    "photo": ((1.78, -1.27, 0.81), (0.27, 0.0, 0.85), 85, 0.175, -0.182),
    "side": ((0.0, -3.2, 0.5), (0.0, 0, 0.45), 50, 0, 0),
    "front": ((3.0, 0.0, 0.6), (0.0, 0, 0.5), 60, 0, 0),
    "close": ((0.95, -0.75, 0.88), (0.54, 0, 0.82), 85, 0, 0),
    "closep": ((1.10, -0.75, 0.80), (0.33, 0.0, 0.82), 85, 0, 0),
    "headside": ((0.53, -1.3, 0.82), (0.53, 0.0, 0.82), 85, 0, 0),
    "headfront": ((1.9, 0.0, 0.84), (0.53, 0.0, 0.82), 85, 0, 0),
    "nose": ((0.85, -0.25, 0.90), (0.64, 0.0, 0.85), 85, 0, 0),
    "top": ((0.45, 0.0, 2.2), (0.45, 0, 0.8), 85, 0, 0),
    "three": ((2.2, -2.4, 1.3), (0.0, 0, 0.45), 45, 0, 0),
    "rear": ((-1.6, -1.6, 0.5), (-0.3, 0, 0.25), 50, 0, 0),
}


def place_photo_camera(o, c):
    """Orbit camera: azimuth from the dog's forward (+X) toward its right side (-Y), elevation, roll."""
    tgt = Vector((c["tx"], c["ty"], c["tz"]))
    az, el = R(c["az"]), R(c["el"])
    o.location = tgt + c["dist"] * Vector((math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), math.sin(el)))
    q = (tgt - o.location).to_track_quat("-Z", "Y")
    o.rotation_euler = (q @ Matrix.Rotation(R(c["roll"]), 4, "Z").to_quaternion()).to_euler()
    o.data.lens = c["lens"]
    o.data.shift_x, o.data.shift_y = c["sx"], c["sy"]
    return o.location.copy(), tgt


def build_camera(coll, kind):
    cam = bpy.data.cameras.new("Camera")
    o = link(bpy.data.objects.new("Camera", cam), coll)
    if kind == "photo":
        loc, tgt = place_photo_camera(o, PARAMS["cam"])
        lens = cam.lens
    else:
        loc, tgt, lens, sx, sy = CAMS[kind]
        o.location = loc
        look_at(o, tgt)
        cam.lens = lens
        cam.shift_x, cam.shift_y = sx, sy
    cam.dof.use_dof = kind in ("photo", "close")
    cam.dof.focus_distance = (Vector(loc) - Vector(tgt)).length - 0.05
    cam.dof.aperture_fstop = 5.6
    bpy.context.scene.camera = o
    return o


# --------------------------------------------------------------------------- pose (photo)
def rot_world(rig, bone, axis, deg):
    """Rotate a pose bone by `deg` about an armature-space axis (given in the bone's rest frame)."""
    pb = rig.pose.bones[bone]
    pb.rotation_mode = "QUATERNION"
    rest = rig.data.bones[bone].matrix_local.to_3x3()
    delta = rest.inverted() @ Matrix.Rotation(R(deg), 3, Vector(axis)) @ rest
    pb.rotation_quaternion = delta.to_quaternion() @ pb.rotation_quaternion


def upd():
    bpy.context.view_layer.update()


def move_bone(rig, bone, pos):
    pb = rig.pose.bones[bone]
    m = pb.matrix.copy()
    m.translation = Vector(pos)
    pb.matrix = m
    upd()


def aim_bone(rig, bone, direction):
    """Point a pose bone's Y axis along an armature-space direction, keeping its head in place."""
    pb = rig.pose.bones[bone]
    m = pb.matrix.copy()
    cur = m.to_3x3() @ Vector((0, 1, 0))
    q = cur.rotation_difference(Vector(direction).normalized())
    rot = q.to_matrix() @ m.to_3x3()
    nm = rot.to_4x4()
    nm.translation = m.translation
    pb.matrix = nm
    upd()


SIT_DROP = 0.40


def apply_photo_pose(rig, pp=None):
    """Sitting pose with head held up and turned, as in the reference photo (angles in PARAMS['pose'])."""
    pp = pp or PARAMS["pose"]
    UP = (0, -1, 0)   # positive = forward end up
    YAW = (0, 0, 1)   # positive = turn toward the dog's left (+Y)
    ROLL = (1, 0, 0)
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    upd()
    move_bone(rig, "root", (0, 0, -SIT_DROP))
    rot_world(rig, "spine_01", UP, pp["spine"])
    rot_world(rig, "spine_02", UP, 2)
    upd()
    for s, sy in (("L", 1), ("R", -1)):
        move_bone(rig, f"front_ik.{s}", (0.22, 0.10 * sy, 0.07))
        move_bone(rig, f"hind_ik.{s}", (-0.30, 0.125 * sy, 0.035))
    upd()
    for s, sy in (("L", 1), ("R", -1)):
        aim_bone(rig, f"front_paw.{s}", (1, 0, -0.35))
        aim_bone(rig, f"hind_foot.{s}", (1, 0.08 * sy, -0.05))
        aim_bone(rig, f"hind_paw.{s}", (1, 0.05 * sy, -0.35))
    aim_bone(rig, "tail_01", (-0.7, 0, -1))
    aim_bone(rig, "tail_02", (-1, 0.05, -0.15))
    aim_bone(rig, "tail_03", (-1, 0.3, 0.0))
    apply_head_pose(rig, pp)


def apply_head_pose(rig, pp):
    """Neck / head / jaw / tongue part of the pose (cheap to re-apply while fitting)."""
    UP, YAW, ROLL = (0, -1, 0), (0, 0, 1), (1, 0, 0)
    for n in ("neck_01", "neck_02", "head", "jaw", "tongue_01", "tongue_02", "tongue_03"):
        rig.pose.bones[n].rotation_quaternion = (1, 0, 0, 0)
    rot_world(rig, "neck_01", UP, pp["neck1"])
    rot_world(rig, "neck_02", UP, pp["neck2"])
    rot_world(rig, "neck_02", YAW, pp["neck_yaw"])
    rot_world(rig, "head", UP, pp["head_pitch"])
    rot_world(rig, "head", YAW, pp["head_yaw"])
    rot_world(rig, "head", ROLL, pp["head_roll"])
    rot_world(rig, "jaw", UP, -pp["jaw"])
    rot_world(rig, "tongue_01", ROLL, 10)
    rot_world(rig, "tongue_01", UP, pp.get("tongue1", 0.0))
    rot_world(rig, "tongue_02", ROLL, pp["tongue_roll"])
    rot_world(rig, "tongue_02", UP, pp.get("tongue2", -4.0))
    rot_world(rig, "tongue_03", UP, pp["tongue_curl"])
    ts = pp.get("tongue_scale", 1.0)
    for n in ("tongue_02", "tongue_03"):
        rig.pose.bones[n].scale = (1.0, ts, 1.0)
    rot_world(rig, "tongue_03", ROLL, 8)
    upd()


# --------------------------------------------------------------------------- render settings
def setup_render(samples, pct):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x = 1480
    sc.render.resolution_y = 816
    sc.render.resolution_percentage = pct
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 8
    try:
        sc.view_settings.view_transform = "Standard"
        sc.view_settings.look = "None"
    except Exception:
        pass
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
    except Exception as e:
        log("GPU setup failed:", e)


# --------------------------------------------------------------------------- main
def main():
    T = time.time()
    reset_scene()
    c_dog = new_collection("Dog")
    c_env = new_collection("Studio")

    body = build_body(c_dog)
    co, nor = mesh_arrays(body.data)
    edges = np.empty(len(body.data.edges) * 2, np.int32)
    body.data.edges.foreach_get("vertices", edges)
    maps = coat_maps(co, nor, edges.reshape(-1, 2))
    set_point_attr(body.data, "skin", "FLOAT_COLOR", maps["skin"])
    set_point_attr(body.data, "coat", "FLOAT_COLOR", maps["color"])
    set_point_attr(body.data, "wet", "FLOAT", maps["wet"])
    set_point_attr(body.data, "fur_length", "FLOAT", maps["length"])
    body.data.materials.append(mat_skin())

    # UVs (needed to attach fur to the surface)
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=R(60), island_margin=0.002)
    bpy.ops.object.mode_set(mode="OBJECT")
    body.data.uv_layers[0].name = "UVMap"
    body.add_rest_position_attribute = True

    eyes = build_eyes(c_dog)
    me_eye = mat_eye()
    for e in eyes:
        e.data.materials.append(me_eye)
    nose = build_nose(c_dog)
    nose.data.materials.append(mat_nose())
    tongue = build_tongue(c_dog)
    tongue.data.materials.append(mat_tongue())
    teeth = build_teeth(c_dog)
    mt = mat_teeth()
    for t in teeth.values():
        t.data.materials.append(mt)
    whisk = build_whiskers(c_dog)
    whisk.data.materials.append(mat_whisker())

    rig = build_rig(c_dog)
    skin_body(body, rig, maps)
    skin_tongue(tongue, rig)
    for o in eyes + [nose, teeth["Upper"], whisk]:
        parent_to_bone(o, rig, "head")
    parent_to_bone(teeth["Lower"], rig, "jaw")

    fur = None
    if OPTS["fur"]:
        fur = build_fur(body, maps, c_dog, OPTS["strands"])
        fur.data.materials.append(mat_fur())
        fur.parent = body
        fur_deform_modifier(fur)

    build_world()
    build_lights(c_env)
    build_camera(c_env, OPTS["cam"])
    setup_render(OPTS["samples"], OPTS["pct"])
    if OPTS["pose"]:
        apply_photo_pose(rig)

    bpy.context.view_layer.update()
    if fur is not None and OPTS["cam"] == "photo" and OPTS["pose"] and not os.environ.get("DOG_NO_PROJECT"):
        project_reference_markings(fur, body, bpy.context.scene.camera, os.path.join(HERE, "reference.png"))
    if OPTS["save"]:
        os.makedirs(os.path.dirname(OPTS["save"]), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=OPTS["save"], compress=True)
        log("saved", OPTS["save"])
    if OPTS["render"]:
        sc = bpy.context.scene
        sc.render.filepath = OPTS["render"]
        if OPTS["border"]:
            sc.render.use_border = True
            sc.render.use_crop_to_border = True
            sc.render.border_min_x, sc.render.border_max_x, sc.render.border_min_y, sc.render.border_max_y = OPTS["border"]
        t = time.time()
        bpy.ops.render.render(write_still=True)
        log(f"rendered {OPTS['render']} in {time.time() - t:.1f}s")
    log(f"total {time.time() - T:.1f}s")


if __name__ == "__main__":
    main()
