"""Offline tests for UV-atlas retexturing (imagegen.uvtex) + a Blender round trip when Blender exists."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from PIL import Image

from imagegen import uvtex
from imagegen.backends import GenerationResult
from imagegen.cli import main as cli_main
from tests.test_imagegen import FakeBackend

BG = (60, 60, 60)


def layout() -> uvtex.UVLayout:
    """Three 64x64 islands on a 256x256 atlas; island 2 is laid out rotated 90 deg, island 3 faces up.

    Island 1 [16:80, 16:80]  wall, upright            normal +Y, up_angle 0
    Island 2 [16:80, 112:176] wall, turned (up -> right)  normal -Y, up_angle 90
    Island 3 [144:208, 16:80] roof, upright           normal +Z, up_angle 0
    One seam: island 1's right edge <-> island 2's left edge.
    """
    ids = np.zeros((256, 256), np.int32)
    ids[16:80, 16:80] = 1
    ids[16:80, 112:176] = 2
    ids[144:208, 16:80] = 3
    normals = np.zeros((256, 256, 3), np.float32)
    normals[ids == 1] = (0, 1, 0)
    normals[ids == 2] = (0, -1, 0)
    normals[ids == 3] = (0, 0, 1)
    up = np.full((256, 256), np.nan, np.float32)
    up[ids == 1] = 0
    up[ids == 2] = 90
    up[ids == 3] = 0
    seams = [{"edge": 0, "islands": [1, 2], "a": [[80, 16], [80, 80]], "b": [[112, 16], [112, 80]], "length_3d": 1.0}]
    uv = uvtex.UVLayout.from_islands(ids, seams)
    uv.normals, uv.up_angle = normals, up
    for isl in uv.info["islands"]:
        isl["texel_density"] = 100.0
    return uv


def flat_atlas(uv: uvtex.UVLayout) -> Image.Image:
    arr = np.full((256, 256, 3), BG, np.uint8)
    arr[uv.islands == 1] = (200, 180, 150)
    arr[uv.islands == 2] = (200, 180, 150)
    arr[uv.islands == 3] = (150, 70, 55)
    return Image.fromarray(arr, "RGB")


def textured(uv: uvtex.UVLayout, base: Image.Image, seed: int = 0) -> Image.Image:
    """A plausible 'restyled' raw output: texture inside islands, background kept."""
    rng = np.random.default_rng(seed)
    b = np.asarray(base).astype(float)
    tex = np.clip(b * 0.7 + 40 + rng.normal(0, 18, b.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(np.where(uv.mask[..., None], tex, np.asarray(base)), "RGB")


# --------------------------------------------------------------------------- #
# layout helpers
# --------------------------------------------------------------------------- #


def test_selectors_and_groups():
    uv = layout()
    assert uvtex.select_texels(uv, "up").sum() == 64 * 64
    assert uvtex.select_texels(uv, "side").sum() == 2 * 64 * 64
    assert uvtex.select_texels(uv, "+y").sum() == 64 * 64
    assert uvtex.select_texels(uv, "islands:2,3").sum() == 2 * 64 * 64
    groups = uvtex.material_groups(uv, [("up", "roof"), ("islands:1", "stone")])
    assert [g["texels"] for g in groups] == [4096, 4096, 4096]
    assert groups[-1]["selector"] == "(unassigned)"  # island 2 isn't covered
    assert uvtex.material_groups(uv, [("up", "roof"), ("rest", "stone")])[-1]["texels"] == 8192
    with pytest.raises(ValueError):
        uvtex.select_texels(uv, "sideways")
    assert uvtex.parse_materials(["up=roof tiles", "side = stone"]) == [("up", "roof tiles"), ("side", "stone")]


def test_rotation_classes_and_resize():
    uv = layout()
    cls = uv.rotation_classes(uv.mask)
    assert set(cls) == {0, 1} and cls[1].sum() == 4096
    big = uv.resized((512, 512))
    assert big.islands.shape == (512, 512) and big.up_angle.shape == (512, 512)
    assert big.info["seams"][0]["a"][0] == [160, 32]


def test_bleed_and_clamp():
    uv = layout()
    base = flat_atlas(uv)
    raw = Image.new("RGB", (256, 256), (10, 200, 10))  # model painted everything green
    out = np.asarray(uvtex.clamp_to_islands(raw, base, uv, padding=6))
    assert (out[uv.mask] == (10, 200, 10)).all()
    assert tuple(out[40, 83]) == (10, 200, 10)  # 3px right of island 1: bled
    assert tuple(out[40, 96]) == BG  # between islands, beyond padding: original


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def check_of(rep, name):
    return next(c for c in rep["checks"] if c["name"] == name)


def test_check_passes_good_result():
    uv = layout()
    base = flat_atlas(uv)
    raw = textured(uv, base)
    rep = uvtex.check(uvtex.clamp_to_islands(raw, base, uv, 8), base, uv, raw=raw, padding=8)
    assert rep["passed"], rep["errors"]
    assert [d["best_shift"] for d in rep["drift"]] == [[0, 0]] * 3


def test_check_flags_drift_only_where_features_matter():
    uv = layout()
    flat = flat_atlas(uv)
    shifted = Image.fromarray(np.roll(np.asarray(textured(uv, flat)), 10, axis=1), "RGB")
    rep = uvtex.check(uvtex.clamp_to_islands(shifted, flat, uv, 8), flat, uv, raw=shifted, padding=8)
    reg = check_of(rep, "layout_registration")
    assert reg["value"]["drifted"] and reg["passed"]  # featureless: clamping handles it
    assert "layout_spill" in rep["warnings"]

    featured = np.asarray(flat).copy()
    featured[20:76:8, 20:76] = (30, 30, 30)  # painted stripes inside island 1 (and similar in 2, 3)
    featured[20:76:8, 116:172] = (30, 30, 30)
    featured[148:204:8, 20:76] = (30, 30, 30)
    featured = Image.fromarray(featured, "RGB")
    moved = Image.fromarray(np.roll(np.asarray(textured(uv, featured)), 10, axis=1), "RGB")
    rep = uvtex.check(uvtex.clamp_to_islands(moved, featured, uv, 8), featured, uv, raw=moved, padding=8)
    assert "layout_registration" in rep["errors"] and rep["repair_islands"]


def test_check_coverage_size_bleed():
    uv = layout()
    base = flat_atlas(uv)
    good = uvtex.clamp_to_islands(textured(uv, base), base, uv, 8)
    holey = np.asarray(good).copy()
    holey[16:80, 16:48] = BG  # half of island 1 left as flat background
    rep = uvtex.check(Image.fromarray(holey), base, uv, padding=8)
    assert "coverage" in rep["errors"] and 1 in rep["repair_islands"]
    # a dark *textured* material near the background colour is not a hole
    rng = np.random.default_rng(1)
    dark = np.asarray(good).copy()
    dark[uv.islands == 3] = np.clip(62 + rng.normal(0, 12, ((uv.islands == 3).sum(), 3)), 0, 255)
    assert check_of(uvtex.check(Image.fromarray(dark), base, uv, padding=8), "coverage")["passed"]
    # no bleed
    no_bleed = np.where(uv.mask[..., None], np.asarray(good), np.asarray(base))
    assert "padding_bleed" in uvtex.check(Image.fromarray(no_bleed), base, uv, padding=8)["errors"]
    # wrong size
    assert "size_match" in uvtex.check(good.resize((128, 128)), base, uv)["errors"]


def test_seam_check_and_fix():
    uv = layout()
    base = flat_atlas(uv)
    arr = np.asarray(uvtex.clamp_to_islands(textured(uv, base), base, uv, 8)).copy()
    arr[uv.islands == 2] = (90, 80, 60)  # island 2 now much darker than island 1 across the seam
    img = uvtex.bleed(Image.fromarray(arr), uv.mask, 8)
    assert not check_of(uvtex.check(img, base, uv, padding=8), "seam_continuity")["passed"]
    fixed = uvtex.fix_seams(img, uv, original=base)
    a, b = np.asarray(fixed, float), np.asarray(img, float)
    # both sides of the seam moved toward each other
    assert np.abs(a[48, 79] - a[48, 112]).mean() < np.abs(b[48, 79] - b[48, 112]).mean() * 0.5
    # far from the seam nothing changed
    assert np.array_equal(a[48, 30], b[48, 30])


def test_flatten_lighting_removes_gradient():
    uv = layout()
    arr = np.asarray(flat_atlas(uv)).astype(float)
    ramp = np.linspace(0.5, 1.3, 64)[None, :, None]
    arr[16:80, 16:80] *= ramp  # island 1 lit from the right
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    out = np.asarray(uvtex.flatten_lighting(img, uv, sigma_frac=0.03, strength=1.0), float)
    before = arr[48, 70].mean() - arr[48, 20].mean()
    after = out[48, 70].mean() - out[48, 20].mean()
    assert abs(after) < abs(before) * 0.5


# --------------------------------------------------------------------------- #
# retexture / repair with a fake model
# --------------------------------------------------------------------------- #


class RecordingBackend(FakeBackend):
    """Returns image 1 with a flat green tint so painted texels are recognisable."""

    def generate(self, prompt, *, references=(), aspect_ratio=None, n=1, **kw):
        self.calls.append({"prompt": prompt, "references": list(references), "aspect_ratio": aspect_ratio})
        img = references[0].convert("RGB") if references else Image.new("RGB", (64, 64))
        arr = np.asarray(img).astype(int)
        arr[..., 1] = np.clip(arr[..., 1] + 60, 0, 255)
        return GenerationResult([Image.fromarray(arr.astype(np.uint8))], "fake", "fake-1")


def test_materials_mode_paints_each_orientation_upright():
    uv = layout()
    base = np.asarray(flat_atlas(uv)).copy()
    base[16:20, 112:176] = (255, 0, 0)  # red strip on island 2's top edge in UV = its left side on the model
    base = Image.fromarray(base)
    fb = RecordingBackend()
    res = uvtex.retexture(base, uv, [Image.new("RGB", (32, 32), "white")], backend=fb,
                          materials=[("up", "roof tiles"), ("side", "stone")], padding=4)  # fmt: skip
    # side splits into two rotation passes (island 1 upright, island 2 turned) + one roof pass
    assert len(fb.calls) == 3
    assert {g["selector"] for g in res.groups} == {"up", "side"}
    # the pass for island 2 saw the atlas turned upright: its red strip (island 2's top edge in UV)
    # arrives as a vertical band, i.e. the model got the island rotated so world-up is up
    def red_band(c):
        a = np.asarray(c["references"][0])
        ys, xs = np.nonzero((a[..., 0] == 255) & (a[..., 1] == 0))
        return (np.ptp(ys) + 1, np.ptp(xs) + 1) if len(xs) else None

    bands = [red_band(c) for c in fb.calls if "stone" in c["prompt"]]
    assert any(b and b[0] > 4 * b[1] for b in bands), bands  # taller than wide
    # later pass of the same material got the first pass as an extra reference
    stone_calls = [c for c in fb.calls if "stone" in c["prompt"]]
    assert sum("same material already painted" in c["prompt"] for c in stone_calls) == 1
    out = np.asarray(res.image)
    orig = np.asarray(base)
    assert (out[uv.mask][:, 1] > orig[uv.mask][:, 1]).all()  # every island texel painted
    assert tuple(out[120, 120]) == BG  # empty space untouched


def test_whole_mode_and_repair_keep_raw(tmp_path):
    uv = layout()
    base = flat_atlas(uv)
    fb = RecordingBackend(out_size=(300, 300))
    res = uvtex.retexture(base, uv, [base], backend=fb, mode="whole")
    assert len(fb.calls) == 1 and len(fb.calls[0]["references"]) == 4  # atlas, outline, orientation, style
    assert "which way the surface under each texel faces" in fb.calls[0]["prompt"]
    final, raw = uvtex.repair(res.image, base, uv, [3], [base], backend=fb, raw=res.raw,
                              materials=[("up", "roof tiles"), ("rest", "stone")])  # fmt: skip
    assert final.size == base.size and raw.size == base.size
    assert "roof tiles" in fb.calls[-1]["prompt"]


def test_cli_check_exit_codes(tmp_path):
    uv = layout()
    d = tmp_path / "uv"
    d.mkdir()
    np.savez_compressed(d / "uv_data.npz", islands=uv.islands, overlap=uv.overlap, normals=uv.normals, up_angle=uv.up_angle)
    (d / "uv_info.json").write_text(json.dumps(uv.info))
    base = flat_atlas(uv)
    base.save(tmp_path / "orig.png")
    uvtex.clamp_to_islands(textured(uv, base), base, uv, 16).save(tmp_path / "good.png")
    assert cli_main(["uv", "check", str(tmp_path / "good.png"), "--original", str(tmp_path / "orig.png"), "--uv", str(d)]) == 0
    base.resize((128, 128)).save(tmp_path / "bad.png")
    assert cli_main(["uv", "check", str(tmp_path / "bad.png"), "--original", str(tmp_path / "orig.png"), "--uv", str(d)]) == 2


# --------------------------------------------------------------------------- #
# Blender round trip
# --------------------------------------------------------------------------- #


def _blender_available() -> bool:
    try:
        from imagegen.blender_bridge import find_blender

        find_blender()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _blender_available(), reason="Blender not installed")
def test_blender_export_and_render(tmp_path):
    from imagegen import blender_bridge

    scene = blender_bridge.make_demo_scene(tmp_path / "demo", size=256)
    blender_bridge.export_uv(scene["blend"], tmp_path / "uv")
    uv = uvtex.UVLayout.load(tmp_path / "uv")
    assert uv.size == (256, 256) and len(uv.island_ids()) >= 4
    assert uv.normals is not None and uv.up_angle is not None
    info = uv.info
    assert info["object"] == "House" and info["textures"][0]["filepath"].endswith("house_atlas.png")
    assert info["sanity"]["overlap_texels"] == 0 and info["seams"]
    # texel counts agree with analytic UV areas
    for isl in info["islands"]:
        assert abs(isl["texels"] - isl["area_px"]) / isl["area_px"] < 0.05
    # every island has a dominant rotation
    assert all(isl["up_rotation"] for isl in info["islands"])
    # the house's walls and roof both exist per texel
    assert uvtex.select_texels(uv, "up").sum() > 0 and uvtex.select_texels(uv, "side").sum() > 0

    uvtex.placeholder(uv).save(scene["atlas"])
    out = blender_bridge.render(scene["blend"], tmp_path / "r", image=scene["atlas"], views="front,iso", res=64, engine="workbench")
    assert len(out["renders"]) == 2 and out["swapped"]
    r = np.asarray(Image.open(out["renders"][0]).convert("RGBA"))
    assert r[..., 3].max() == 255  # the object is in frame
    shutil.rmtree(tmp_path / "demo")


# --------------------------------------------------------------------------- #
# offset paint (found in a real run: the model shifted a wall pass ~12 px every time)
# --------------------------------------------------------------------------- #


class ShiftingBackend(FakeBackend):
    """Paints the highlighted (masked) area green, but displaced by ``shift`` px — like a real model."""

    def __init__(self, shift=(10, 0), **kw):
        super().__init__(**kw)
        self.shift = shift

    def generate(self, prompt, *, references=(), aspect_ratio=None, n=1, **kw):
        self.calls.append({"prompt": prompt, "references": list(references)})
        crop = np.asarray(references[0].convert("RGB")).astype(int)
        marked = np.abs(np.asarray(references[1].convert("RGB")).astype(int) - crop).max(axis=2) > 20
        dx, dy = self.shift
        from scipy import ndimage

        moved = ndimage.shift(marked.astype(np.uint8), (dy, dx), order=0, cval=0).astype(bool)  # no wrap-around
        out = crop.copy()
        rng = np.random.default_rng(len(self.calls))
        out[moved] = np.clip(np.array([40, 170, 60]) + rng.normal(0, 12, (moved.sum(), 3)), 0, 255)
        return GenerationResult([Image.fromarray(out.astype(np.uint8))], "fake", "fake-1")


def test_register_patch_finds_offset():
    rng = np.random.default_rng(0)
    before = rng.integers(0, 60, (128, 160, 3)).astype(np.uint8)
    mask = np.zeros((128, 160), bool)
    mask[30:90, 40:120] = True
    after = before.copy()
    after[np.roll(mask, (-5, 12), axis=(0, 1))] = (200, 30, 30)
    dx, dy, gain = uvtex.register_patch(before, after, mask)
    assert (dx, dy) == (12, -5) and gain > 1.2
    after2 = before.copy()
    after2[mask] = (200, 30, 30)
    assert uvtex.register_patch(before, after2, mask)[:2] == (0, 0)


def test_offset_paint_is_corrected_and_leaves_no_strip():
    uv = layout()
    base = flat_atlas(uv)
    fb = ShiftingBackend(shift=(10, 0))
    res = uvtex.retexture(base, uv, [base], backend=fb, materials=[("up", "roof"), ("rest", "stone")], padding=4)
    corrected = [c.get("shift_corrected") for c in res.calls]
    assert all(c is not None and abs(c[0]) + abs(c[1]) >= 8 for c in corrected), corrected
    holes = uvtex.unpainted(res.image, base, uv, [("up", "roof"), ("rest", "stone")])
    assert holes.sum() / uv.mask.sum() < 0.01
    rep = uvtex.check(res.image, base, uv, raw=res.raw, padding=4, materials=[("up", "roof"), ("rest", "stone")])
    assert "coverage" not in rep["errors"], rep["checks"]


def test_unpainted_strip_detected_per_island_and_filled():
    uv = layout()
    base = flat_atlas(uv)
    good = np.asarray(uvtex.clamp_to_islands(textured(uv, base), base, uv, 8)).copy()
    good[16:80, 70:80] = np.asarray(base)[16:80, 70:80]  # 10 px strip of island 1 still the old flat colour
    img = Image.fromarray(good)
    holes = uvtex.unpainted(img, base, uv)
    assert holes[16:80, 72:78].all() and not holes[16:80, 20:60].any()
    rep = uvtex.check(img, base, uv, padding=8)
    assert "coverage" in rep["errors"] and rep["repair_islands"] == [1]  # 15% of island 1, only 5% overall
    fixed, info = uvtex.finish(img, base, uv, padding=8, max_fill=0.2)
    assert 1 in info["filled_islands"]
    assert "coverage" not in uvtex.check(fixed, base, uv, padding=8)["errors"]
