"""Offline tests for imagegen: pixel ops, pipeline, atlas projects, and both backends (faked)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import stat
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from imagegen import atlas, ops, pipeline
from imagegen.backends import CodexBackend, GenerationResult, ImageGenError, OpenRouterBackend
from imagegen.cli import main as cli_main

ASSETS = Path(__file__).resolve().parent / "assets"
DOG = ASSETS / "dog-1.png"


class FakeBackend:
    """Deterministic stand-in for a model: 'edits' by tinting the first reference, 'generates' noise."""

    name = "fake"
    model = "fake-1"
    aspect_ratios = ["1:1", "3:2", "2:3"]

    def __init__(self, out_size=(300, 300), judge_fn=None):
        self.calls: list[dict] = []
        self.judge_calls: list[dict] = []
        self.out_size = out_size
        self.judge_fn = judge_fn or (lambda prompt, images: {"score": 9, "matches": True, "issues": [], "fix_instruction": ""})

    def generate(self, prompt, *, references=(), aspect_ratio=None, n=1, **options):
        self.calls.append({"prompt": prompt, "references": list(references), "aspect_ratio": aspect_ratio})
        if references:
            base = references[0].convert("RGB").resize(self.out_size)
            arr = np.asarray(base).astype(np.int16)
            arr[..., 0] = np.clip(arr[..., 0] + 40, 0, 255)  # visible change everywhere
            img = Image.fromarray(arr.astype(np.uint8), "RGB")
        else:
            rng = np.random.default_rng(len(self.calls))
            img = Image.fromarray(rng.integers(0, 255, (*self.out_size[::-1], 3), dtype=np.uint8), "RGB")
        return GenerationResult(images=[img] * n, backend=self.name, model=self.model, usage={"calls": 1})

    def judge(self, prompt, images, schema):
        self.judge_calls.append({"prompt": prompt, "images": list(images)})
        return self.judge_fn(prompt, images)


def gradient(w=64, h=48) -> Image.Image:
    x = np.linspace(0, 255, w)[None, :].repeat(h, 0)
    y = np.linspace(0, 255, h)[:, None].repeat(w, 1)
    return Image.fromarray(np.dstack([x, y, 255 - x]).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- #
# ops
# --------------------------------------------------------------------------- #


def test_parse_helpers():
    assert ops.parse_size("256") == (256, 256)
    assert ops.parse_size("512x128") == (512, 128)
    assert ops.parse_grid("4x2") == (4, 2)
    assert ops.parse_region("10,20,30,40", (100, 100)) == (10, 20, 40, 60)
    assert ops.parse_region("0.5,0.0,0.5,0.25", (200, 100)) == (100, 0, 200, 25)
    assert ops.parse_region("90,90,50,50", (100, 100)) == (90, 90, 100, 100)  # clamped
    assert ops.parse_color("#ff00ff") == (255, 0, 255)
    assert ops.parse_color("f0f") == (255, 0, 255)
    assert ops.parse_color("1,2,3") == (1, 2, 3)
    with pytest.raises(ValueError):
        ops.parse_region("1,2,3", (10, 10))


def test_aspect_helpers():
    assert ops.nearest_aspect(1920, 1080, ["1:1", "3:2", "16:9"]) == "16:9"
    assert ops.nearest_aspect(100, 300, ["1:1", "3:2", "2:3"]) == "2:3"
    box = ops.fit_box_to_aspect((40, 40, 60, 50), 1.0, (100, 100))  # 20x10 -> square
    assert box[2] - box[0] == box[3] - box[1] == 20
    edge = ops.fit_box_to_aspect((0, 0, 10, 80), 1.0, (50, 100))  # can't exceed image width
    assert edge[0] >= 0 and edge[2] <= 50


def test_fit_modes():
    img = gradient(200, 100)
    assert ops.fit(img, (50, 50), "cover").size == (50, 50)
    assert ops.fit(img, (50, 50), "contain").size == (50, 50)
    assert ops.fit(img, (50, 80), "stretch").size == (50, 80)


def test_seamless_blend_fixes_seams_and_keeps_centre():
    img = gradient(128, 128)  # worst case: opposite edges maximally different
    assert ops.seam_score(img) > 10
    out = ops.seamless_blend(img, band=0.25)
    assert ops.seam_score(out) < 1.6
    a, b = np.asarray(img), np.asarray(out)
    assert np.array_equal(a[40:88, 40:88], b[40:88, 40:88])  # centre untouched


def test_roll_and_cross_mask():
    img = gradient(40, 20)
    assert np.array_equal(np.asarray(ops.roll(ops.roll(img, 7, 3), -7, -3)), np.asarray(img))
    m = np.asarray(ops.cross_mask((100, 100), 0.2))
    assert m[50, 0] == 255 and m[0, 50] == 255 and m[0, 0] == 0


def test_tiles_cover_and_blend_identity():
    size = (300, 200)
    boxes = ops.tile_boxes(size, 128, 32)
    cover = np.zeros(size[::-1], bool)
    for l, t, r, b in boxes:
        cover[t:b, l:r] = True
    assert cover.all()
    img = gradient(*size)
    out = ops.blend_tiles(size, [(b, img.crop(b)) for b in boxes])
    assert np.abs(np.asarray(out, int) - np.asarray(img, int)).max() <= 1


def test_chroma_key():
    img = Image.new("RGB", (20, 20), (255, 0, 255))
    img.paste((0, 128, 0), (5, 5, 15, 15))
    out = ops.chroma_key(img)
    a = np.asarray(out)
    assert out.mode == "RGBA"
    assert a[0, 0, 3] == 0 and a[10, 10, 3] == 255
    assert tuple(a[10, 10, :3]) == (0, 128, 0)


def test_pack_slice_roundtrip_and_extrude():
    cells = [Image.new("RGBA", (16, 16), (i * 40, 0, 0, 255)) for i in range(5)]
    atlas_img, frames = ops.pack_grid(cells, 3, (16, 16), padding=2)
    assert atlas_img.size == (3 * 20, 2 * 20)
    x, y, w, h = frames[4]
    assert (x, y) == (1 * 20 + 2, 1 * 20 + 2)
    assert atlas_img.getpixel((x, y))[:3] == (160, 0, 0)
    assert atlas_img.getpixel((x - 2, y - 2))[:3] == (160, 0, 0)  # extruded padding
    pot, _ = ops.pack_grid(cells, 3, (16, 16), padding=2, power_of_two=True)
    assert pot.size == (64, 64)
    unpadded, _ = ops.pack_grid(cells[:4], 2, (16, 16))
    back = ops.slice_grid(unpadded, 2, 2)
    assert [b.getpixel((0, 0)) for b in back] == [c.getpixel((0, 0)) for c in cells[:4]]


def test_grid_guide_box():
    img, box = ops.grid_guide(4, 2, (64, 64), ["a", "b"], canvas=(300, 300))
    assert img.size == (300, 300)
    assert box[2] - box[0] == 256 and box[3] - box[1] == 128


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


def test_generate_resizes_to_requested_size():
    fb = FakeBackend()
    images, res = pipeline.generate("a cat", backend=fb, size=(64, 32))
    assert images[0].size == (64, 32)
    assert fb.calls[0]["aspect_ratio"] == "3:2"


def test_region_edit_never_touches_pixels_outside_region():
    base = ops.load_image(DOG).convert("RGB")
    fb = FakeBackend(out_size=(512, 512))
    region = (300, 100, 420, 220)
    res = pipeline.edit(base, "make it red", backend=fb, region=region, feather=4, color_match=False)
    out, before = np.asarray(res.image), np.asarray(base)
    assert res.image.size == base.size
    # Everything outside the region is bit-identical (feathering only ramps inward).
    keep = np.ones(before.shape[:2], bool)
    keep[region[1] : region[3], region[0] : region[2]] = False
    assert np.array_equal(out[keep], before[keep])
    # ...and the inside changed.
    assert np.abs(out[140:180, 340:380].astype(int) - before[140:180, 340:380]).mean() > 5
    # The model was shown the context crop plus a highlighted copy, at a supported aspect.
    call = fb.calls[0]
    assert len(call["references"]) == 2 and call["aspect_ratio"] in fb.aspect_ratios
    assert res.context_box[0] <= region[0] and res.context_box[2] >= region[2]


def test_color_match_cancels_global_model_drift():
    base = ops.load_image(DOG).convert("RGB")
    # The fake shifts red by +40 everywhere, i.e. pure drift: matching on the ring should undo it.
    res = pipeline.edit(base, "x", backend=FakeBackend(out_size=(512, 512)), region=(300, 100, 420, 220))
    diff = np.abs(np.asarray(res.image, int) - np.asarray(base, int))[140:180, 340:380]
    assert diff.mean() < 8


def test_mask_edit_uses_mask_bbox():
    base = gradient(100, 100)
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (10, 10, 30, 30))
    res = pipeline.edit(base, "x", backend=FakeBackend(), mask=mask, feather=0, color_match=False)
    assert res.region == (10, 10, 30, 30)
    assert np.array_equal(np.asarray(res.image)[50:, 50:], np.asarray(base)[50:, 50:])


def test_full_edit_keeps_size():
    base = gradient(120, 80)
    res = pipeline.edit(base, "x", backend=FakeBackend(out_size=(300, 200)))
    assert res.image.size == (120, 80)


def test_upscale_local_and_ai():
    img = gradient(50, 40)
    assert pipeline.upscale(img, 2).size == (100, 80)
    fb = FakeBackend()
    out = pipeline.upscale(img, 4, method="ai", backend=fb, tile=128, overlap=32)
    assert out.size == (200, 160)
    assert len(fb.calls) == len(ops.tile_boxes((200, 160), 128, 32)) > 1


def test_seamless_ai_repaints_seams_into_the_interior():
    class Smoother(FakeBackend):
        """Stands in for a model that repaints the seam cross smoothly (here: a heavy blur)."""

        def generate(self, prompt, *, references=(), **kw):
            self.calls.append({"prompt": prompt, "references": list(references)})
            from PIL import ImageFilter

            return GenerationResult([references[0].filter(ImageFilter.GaussianBlur(12))], "fake", "fake-1")

    img = gradient(96, 96)
    fb = Smoother()
    out = pipeline.seamless(img, method="ai", backend=fb, band=0.3)
    assert out.size == img.size and len(fb.calls) == 1
    assert "seams now run through the middle" in fb.calls[0]["prompt"]
    # The offset put the old border seam in the repainted cross; offsetting back puts that
    # (now continuous) content on the borders, so the texture wraps.
    assert ops.seam_score(out) < ops.seam_score(img) / 5
    # The middle of each quadrant (outside the cross, away from the feather) is untouched.
    a, b = np.asarray(out), np.asarray(img)
    assert np.array_equal(a[20:28, 20:28], b[20:28, 20:28])


# --------------------------------------------------------------------------- #
# atlas projects
# --------------------------------------------------------------------------- #


def small_spec(**kw) -> atlas.AtlasSpec:
    data = {
        "name": "t",
        "cell_size": 32,
        "columns": 2,
        "padding": 1,
        "style": "test style",
        "tileable": True,
        "cells": [{"name": "a", "prompt": "grass"}, {"name": "b", "prompt": "dirt"}, {"name": "c", "prompt": "rock"}],
        **kw,
    }
    return atlas.AtlasSpec.from_dict(data)


def test_spec_validation():
    with pytest.raises(ValueError, match="Duplicate"):
        atlas.AtlasSpec.from_dict({"cells": [{"name": "a"}, {"name": "a"}]})
    with pytest.raises(ValueError, match="Unknown"):
        atlas.AtlasSpec.from_dict({"cells": [{"name": "a"}], "bogus": 1})
    with pytest.raises(ValueError):
        atlas.AtlasSpec.from_dict({"cells": [{"name": "bad name"}]})
    assert small_spec().grid == (2, 2)


def test_build_chains_style_and_is_resumable(tmp_path):
    fb = FakeBackend()
    res = atlas.build(small_spec(), tmp_path / "proj", fb, concurrency=2)
    assert res["errors"] == {} and res["missing"] == []
    assert len(fb.calls) == 3
    assert len(fb.calls[0]["references"]) == 0  # style anchor
    assert all(len(c["references"]) == 1 for c in fb.calls[1:])  # others match the anchor
    assert "Seamless tileable" in fb.calls[0]["prompt"] and "test style" in fb.calls[0]["prompt"]

    meta = json.loads((tmp_path / "proj" / "atlas.json").read_text())
    assert set(meta["frames"]) == {"a", "b", "c"}
    f = meta["frames"]["c"]["frame"]
    assert (f["x"], f["y"], f["w"], f["h"]) == (1, 35, 32, 32)
    assert Image.open(tmp_path / "proj" / "atlas.png").size == (68, 68)
    for n in "abc":
        cell = Image.open(tmp_path / "proj" / "cells" / f"{n}.png")
        assert cell.size == (32, 32)
        assert ops.seam_score(cell) <= atlas.SEAM_THRESHOLD  # auto_seamless on noise output

    atlas.build(atlas.AtlasSpec.load(tmp_path / "proj" / "spec.json"), tmp_path / "proj", fb)
    assert len(fb.calls) == 3  # nothing regenerated


def test_build_sheet_mode(tmp_path):
    fb = FakeBackend(out_size=(600, 600))
    res = atlas.build(small_spec(mode="sheet"), tmp_path / "p", fb)
    assert len(fb.calls) == 1
    assert "2 x 2 grid" in fb.calls[0]["prompt"]
    assert res["missing"] == [] and (tmp_path / "p" / "sheet" / "guide.png").is_file()


def test_transparent_sprites_are_keyed(tmp_path):
    class KeyBackend(FakeBackend):
        def generate(self, prompt, **kw):
            img = Image.new("RGB", (100, 100), (255, 0, 255))
            img.paste((10, 200, 10), (30, 30, 70, 70))
            self.calls.append({"prompt": prompt, **kw})
            return GenerationResult([img], "fake", "fake-1")

    spec = small_spec(tileable=False, background="transparent")
    atlas.build(spec, tmp_path / "s", KeyBackend())
    cell = Image.open(tmp_path / "s" / "cells" / "a.png")
    assert cell.mode == "RGBA" and cell.getpixel((0, 0))[3] == 0 and cell.getpixel((16, 16))[3] == 255


def test_edit_cell_history_and_revert(tmp_path):
    proj = tmp_path / "proj"
    atlas.build(small_spec(), proj, FakeBackend())
    original = np.asarray(Image.open(proj / "cells" / "b.png"))
    fb = FakeBackend()
    res = atlas.edit_cell(proj, "b", fb, instruction="add cracks", region=(0, 0, 16, 16))
    assert res["history"] == 1
    assert "add cracks" in fb.calls[0]["prompt"] and "tileable" in fb.calls[0]["prompt"]
    edited = np.asarray(Image.open(proj / "cells" / "b.png"))
    assert not np.array_equal(edited, original)

    atlas.edit_cell(proj, "b", fb, prompt="wet mud")
    assert atlas.AtlasSpec.load(proj / "spec.json").cell("b").prompt == "wet mud"

    atlas.revert_cell(proj, "b", steps=2)
    assert np.array_equal(np.asarray(Image.open(proj / "cells" / "b.png")), original)
    log = [json.loads(line)["op"] for line in (proj / "log.jsonl").read_text().splitlines()]
    assert log[-3:] == ["edit", "regenerate", "revert"]

    with pytest.raises(ValueError):
        atlas.edit_cell(proj, "b", fb, instruction="x", prompt="y")
    with pytest.raises(KeyError):
        atlas.edit_cell(proj, "nope", fb, instruction="x")


def test_review_fix_edits_failing_cells(tmp_path):
    proj = tmp_path / "proj"
    atlas.build(small_spec(), proj, FakeBackend())

    def judge(prompt, images):
        if "Cell name: b" in prompt:
            return {"score": 5, "matches": False, "issues": ["too bright"], "fix_instruction": "darken it"}
        if "Cell name: c" in prompt:
            return {"score": 2, "matches": False, "issues": ["wrong subject"], "fix_instruction": "redo"}
        if "texture atlas, cells numbered" in prompt:
            return {"outliers": [{"name": "a", "reason": "different palette"}, {"name": "zzz", "reason": "ignored"}]}
        return {"score": 9, "matches": True, "issues": [], "fix_instruction": ""}

    fb = FakeBackend(judge_fn=judge)
    rep = atlas.review(proj, fb, fix=True, rounds=1, concurrency=1)
    assert rep["rounds"][0]["failing"] == ["b", "c"]
    assert rep["style_outliers"] == [{"name": "a", "reason": "different palette"}]
    prompts = [c["prompt"] for c in fb.calls]
    assert any("darken it" in p for p in prompts)  # b edited
    assert any(p.startswith("Art style") and "rock" in p for p in prompts)  # c regenerated
    assert any("Restyle" in p for p in prompts)  # a restyled against another cell
    assert json.loads((proj / "review.json").read_text())["cells"]["b"]["score"] == 5
    assert len(fb.judge_calls[0]["images"]) == 2  # tileable: cell + 2x2 preview


def test_import_and_upscale(tmp_path):
    src = ops.pack_grid([gradient(20, 20), gradient(20, 20).rotate(90), gradient(20, 20), gradient(20, 20)], 2, (20, 20), padding=3)[0]
    res = atlas.import_atlas(src, (2, 2), tmp_path / "imp", names=["n1", "n2", "n3", "n4"], source_padding=3)
    assert res["cells"] == 4 and res["size"] == [40, 40]
    cell = Image.open(tmp_path / "imp" / "cells" / "n2.png")
    assert cell.size == (20, 20)
    up = atlas.upscale_atlas(tmp_path / "imp", 2)
    assert up["size"] == [80, 80]
    assert atlas.AtlasSpec.load(tmp_path / "imp" / "spec.json").cell_size == (40, 40)


def test_pack_images_writes_json(tmp_path):
    files = []
    for i in range(3):
        p = tmp_path / f"tile{i}.png"
        Image.new("RGB", (8, 8), (i * 50, 0, 0)).save(p)
        files.append(p)
    res = atlas.pack_images(files, tmp_path / "out.png", columns=3, padding=1)
    meta = json.loads((tmp_path / "out.json").read_text())
    assert res["size"] == [30, 10] and list(meta["frames"]) == ["tile0", "tile1", "tile2"]
    assert meta["frames"]["tile1"]["uv"] == [11 / 30, 1 / 10, 19 / 30, 9 / 10]


# --------------------------------------------------------------------------- #
# Codex backend (fake codex executable)
# --------------------------------------------------------------------------- #

FAKE_CODEX = r'''#!{python}
import json, os, sys, pathlib
args = sys.argv[1:]
prompt = sys.stdin.read()
log = pathlib.Path(os.environ["FAKE_CODEX_LOG"])
log.write_text(json.dumps({{"args": args, "prompt": prompt}}))
mode = os.environ.get("FAKE_CODEX_MODE", "ok")
print(json.dumps({{"type": "thread.started", "thread_id": "thread-123"}}))
if mode == "old":
    print(json.dumps({{"type": "turn.failed", "error": {{"message": "The 'gpt-x' model requires a newer version of Codex."}}}}))
    sys.exit(1)
if "--output-schema" in args:
    out = args[args.index("-o") + 1]
    pathlib.Path(out).write_text(json.dumps({{"score": 8, "matches": True, "issues": [], "fix_instruction": ""}}))
elif mode != "noimage":
    from PIL import Image
    d = pathlib.Path(os.environ["FAKE_IMAGES_ROOT"]) / "thread-123"
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (1, 2, 3)).save(d / "ig_1.png")
print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "DONE"}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 5}}}}))
'''


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    script = tmp_path / "codex"
    script.write_text(FAKE_CODEX.format(python=sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(tmp_path / "call.json"))
    monkeypatch.setenv("FAKE_IMAGES_ROOT", str(tmp_path / "gen"))
    backend = CodexBackend(model="gpt-test", codex_bin=str(script), images_root=tmp_path / "gen")
    return backend, tmp_path


def test_codex_generate_collects_image_and_passes_refs(fake_codex):
    backend, tmp = fake_codex
    res = backend.generate("a mossy rock", references=[gradient(10, 10)], aspect_ratio="3:2")
    assert res.image.size == (40, 40) and res.meta["thread_id"] == "thread-123"
    call = json.loads((tmp / "call.json").read_text())
    args = call["args"]
    assert args[:2] == ["exec", "--json"] and args[args.index("-m") + 1] == "gpt-test"
    assert "--ignore-user-config" in args and "--ephemeral" in args
    assert args.index("-") < args.index("-i")  # prompt from stdin, before variadic -i
    assert "a mossy rock" in call["prompt"] and "exactly 1 time" in call["prompt"] and "landscape" in call["prompt"]


def test_codex_judge_uses_output_schema(fake_codex):
    backend, tmp = fake_codex
    verdict = backend.judge("rate it", [gradient()], atlas.REVIEW_SCHEMA)
    assert verdict["score"] == 8
    assert "--output-schema" in json.loads((tmp / "call.json").read_text())["args"]


def test_codex_errors(fake_codex, monkeypatch):
    backend, _ = fake_codex
    monkeypatch.setenv("FAKE_CODEX_MODE", "old")
    with pytest.raises(ImageGenError, match="codex update"):
        backend.generate("x")
    monkeypatch.setenv("FAKE_CODEX_MODE", "noimage")
    with pytest.raises(ImageGenError, match="without producing an image"):
        backend.generate("x")


# --------------------------------------------------------------------------- #
# OpenRouter backend (mocked HTTP)
# --------------------------------------------------------------------------- #


def _png_b64(color=(9, 9, 9)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_openrouter_generate_and_judge():
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        seen.append((request.url.path, body))
        if request.url.path.endswith("/images/models"):
            return httpx.Response(200, json={"data": [{"id": "m/img", "supported_parameters": {
                "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9", "auto"]},
                "input_references": {"type": "range", "min": 0, "max": 4}}}]})  # fmt: skip
        if request.url.path.endswith("/images"):
            return httpx.Response(200, json={"data": [{"b64_json": _png_b64()}], "usage": {"cost": 0.01}})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"score": 6, "matches": false, "issues": [], "fix_instruction": "x"}'}}]})
        return httpx.Response(404)

    b = OpenRouterBackend(model="m/img", api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    res = b.generate("tex", references=[gradient()], aspect_ratio="3:2", resolution="2K", seed=4)
    assert res.image.size == (16, 16) and res.usage["cost"] == 0.01
    _, body = next(s for s in seen if s[0].endswith("/images"))
    assert body["aspect_ratio"] == "16:9"  # nearest supported to 3:2
    assert "resolution" not in body and "seed" not in body  # unsupported -> dropped
    assert body["input_references"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert b.judge("rate", [gradient()], atlas.REVIEW_SCHEMA)["score"] == 6


def test_openrouter_needs_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ImageGenError, match="OPENROUTER_API_KEY"):
        OpenRouterBackend()


# --------------------------------------------------------------------------- #
# CLI + MCP
# --------------------------------------------------------------------------- #


def test_cli_offline_commands(tmp_path, capsys):
    grad = tmp_path / "g.png"
    gradient(64, 64).save(grad)
    assert cli_main(["seamless", str(grad), "-o", str(tmp_path / "s.png"), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["seam_score_after"] < out["seam_score_before"]
    assert cli_main(["atlas", "init", str(tmp_path / "spec.json")]) == 0
    assert atlas.AtlasSpec.load(tmp_path / "spec.json").cells
    assert cli_main(["atlas", "guide", "--grid", "3x2", "--cell", "32", "-o", str(tmp_path / "guide.png")]) == 0
    assert cli_main(["upscale", str(grad), "-o", str(tmp_path / "u.png"), "-s", "1.5"]) == 0
    assert Image.open(tmp_path / "u.png").size == (96, 96)
    assert cli_main(["atlas", "revert", str(tmp_path / "missing"), "a"]) == 1  # clean error, no traceback


def test_mcp_server_lists_tools():
    from imagegen.mcp_server import server

    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"generate_image", "edit_image", "upscale_image", "make_seamless", "atlas_build",
            "atlas_edit_cell", "atlas_review", "atlas_revert_cell", "atlas_import"} <= names  # fmt: skip


def test_mcp_image_tool_returns_text_and_preview(tmp_path):
    from imagegen.mcp_server import server

    src = tmp_path / "k.png"
    Image.new("RGB", (8, 8), (255, 0, 255)).save(src)
    res = asyncio.run(server.call_tool("chroma_key", {"image": str(src), "out": str(tmp_path / "o.png")}))
    content = res if isinstance(res, (list, tuple)) else res.content
    assert [type(c).__name__ for c in content] == ["TextContent", "ImageContent"]
    assert Image.open(tmp_path / "o.png").mode == "RGBA"
