"""Offline tests for render_eval.

Synthetic RGBA shapes stand in for renders (their alpha channel is the mask, so no
segmentation model is needed). OpenRouter calls are replaced with fakes. The
model-backed evals (LPIPS, depth, normals) only run with RENDER_EVAL_SLOW_TESTS=1.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from render_eval import EvalConfig, composite_score, make_pair, run_evals
from render_eval import color, edges, embedding, judge, silhouette
from render_eval._geometry import boundary_prf
from render_eval.base import EvalResult, EvalSkipped

slow = pytest.mark.skipif(os.getenv("RENDER_EVAL_SLOW_TESTS") != "1", reason="set RENDER_EVAL_SLOW_TESTS=1 to run model-backed evals")


def shape(kind: str = "ellipse", size=(320, 240), box=(80, 50, 240, 200), fill=(200, 60, 40), stripe=True) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    getattr(d, kind)(box, fill=(*fill, 255))
    if stripe:  # some interior structure for the edge eval
        x0, y0, x1, y1 = box
        d.rectangle((x0 + (x1 - x0) // 3, y0 + 10, x0 + (x1 - x0) // 3 + 12, y1 - 10), fill=(30, 30, 160, 255))
    return img


CFG = EvalConfig(size=256)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_alpha_is_used_as_mask():
    pair = make_pair(shape(), shape(), CFG)
    assert pair.meta["reference"]["mask_source"] == "alpha"
    assert pair.ref.shape == (256, 256, 3) and pair.ref_mask.shape == (256, 256)
    assert 0.2 < pair.ref_mask.mean() < 0.9


def test_bbox_alignment_removes_translation_and_scale():
    small_shifted = shape(size=(640, 480), box=(20, 300, 100, 394))  # same aspect, different place and size
    big = shape(size=(640, 480), box=(200, 40, 520, 416))
    pair = make_pair(small_shifted, big, CFG)
    r = silhouette.evaluate(pair, CFG)
    assert r.score > 0.95, r.metrics


def test_align_none_keeps_framing_differences():
    a = shape(size=(640, 480), box=(20, 300, 100, 394))
    b = shape(size=(640, 480), box=(400, 40, 560, 228))
    r = silhouette.evaluate(make_pair(a, b, EvalConfig(size=256, align="none")), CFG)
    assert r.score < 0.05


def test_explicit_mask_overrides_alpha():
    rgb = shape().convert("RGB")  # no alpha -> would need segmentation
    mask = Image.new("L", rgb.size, 0)
    ImageDraw.Draw(mask).ellipse((80, 50, 240, 200), fill=255)
    pair = make_pair(shape(), rgb, CFG, render_mask=mask)
    assert pair.meta["render"]["mask_source"] == "provided"
    assert silhouette.evaluate(pair, CFG).score > 0.97


# --------------------------------------------------------------------------- #
# Geometry evals
# --------------------------------------------------------------------------- #


def test_silhouette_identical_and_different():
    same = silhouette.evaluate(make_pair(shape(), shape(), CFG), CFG)
    assert same.score == pytest.approx(1.0)
    assert same.metrics["boundary_f1"] == pytest.approx(1.0)
    tall = shape(box=(140, 10, 180, 230))  # much narrower object
    diff = silhouette.evaluate(make_pair(shape(), tall, CFG), CFG)
    assert diff.score < 0.6
    assert diff.metrics["aspect_ratio_error"] > 1.0


def test_silhouette_debug_artifact():
    r = silhouette.evaluate(make_pair(shape(), shape("rectangle"), EvalConfig(size=128, debug=True)), EvalConfig(debug=True))
    assert r.artifacts["silhouette"].shape == (128, 128, 3)


def test_boundary_prf_tolerance():
    a = np.zeros((64, 64), bool)
    a[32, 10:50] = True
    b = np.roll(a, 2, axis=0)  # 2 px shift
    assert boundary_prf(b, a, tol=3)["f1"] == pytest.approx(1.0)
    assert boundary_prf(b, a, tol=1)["f1"] == pytest.approx(0.0)
    assert boundary_prf(np.zeros_like(a), a, tol=3)["f1"] == 0.0


def test_edges_identical_vs_missing_structure():
    same = edges.evaluate(make_pair(shape(), shape(), CFG), CFG)
    assert same.score == pytest.approx(1.0)
    no_stripe = edges.evaluate(make_pair(shape(), shape(stripe=False), CFG), CFG)
    assert no_stripe.metrics["recall"] < 0.9  # the stripe's edges are missing from the render
    assert no_stripe.metrics["interior_f1"] < 0.5


# --------------------------------------------------------------------------- #
# Color
# --------------------------------------------------------------------------- #


def test_color_identical_and_shifted():
    same = color.evaluate(make_pair(shape(), shape(), CFG), CFG)
    assert same.score == pytest.approx(1.0, abs=1e-6)
    blue = color.evaluate(make_pair(shape(), shape(fill=(40, 60, 200)), CFG), CFG)
    assert blue.score < 0.3
    assert blue.metrics["mean_delta_e2000"] > 20
    assert blue.details["render_palette"][0]["hex"].startswith("#")


def test_color_ignores_layout_changes():
    # A mirrored copy has exactly the same colors, so the shared palette must see no difference.
    from PIL import ImageOps

    multi = shape(box=(40, 30, 280, 210))
    ImageDraw.Draw(multi).ellipse((60, 60, 150, 150), fill=(240, 230, 220, 255))
    r = color.evaluate(make_pair(multi, ImageOps.mirror(multi), CFG), CFG)
    assert r.score > 0.97, r.metrics


def test_emd_matches_hand_computed_value():
    c1 = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    c2 = np.array([[0.0, 0, 0]])
    assert color.emd(c1, np.array([0.5, 0.5]), c2, np.array([1.0])) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# OpenRouter-backed evals with fakes
# --------------------------------------------------------------------------- #


class _Res(list):
    """Stands in for render_eval.openrouter.EmbeddingResult: indexable vectors plus model/usage/dimensions."""

    model = "fake/embed"
    usage = {"prompt_tokens": 2}

    @property
    def dimensions(self):
        return len(self[0])


def test_embedding_eval_uses_cosine_and_sends_both_images():
    class E:
        def __init__(self):
            self.images = None

        def embed_images(self, images):
            self.images = images
            return _Res([[1.0, 0.0], [0.6, 0.8]])

    e = E()
    cfg = EvalConfig(size=128, keep_vectors=True)
    r = embedding.evaluate(make_pair(shape(), shape(), cfg), cfg, embedder=e)
    assert r.score == pytest.approx(0.6)
    assert len(e.images) == 2 and all(isinstance(i, Image.Image) for i in e.images)
    assert r.details["render_vector"] == [0.6, 0.8]


def test_embedding_eval_skips_without_key(monkeypatch):
    import render_eval.openrouter as he

    def boom(*a, **k):
        raise he.EmbeddingError("OPENROUTER_API_KEY is not set")

    monkeypatch.setattr(he, "OpenRouterEmbedder", boom)
    with pytest.raises(EvalSkipped):
        embedding.evaluate(make_pair(shape(), shape(), CFG), CFG)


class FakeChat:
    """Mimics openai.OpenAI().chat.completions.create and records requests."""

    def __init__(self, replies, reject_schema=False):
        self.replies = list(replies)
        self.requests = []
        self.reject_schema = reject_schema
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.reject_schema and "response_format" in kwargs:
            raise ValueError("response_format json_schema not supported")
        content = self.replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            model=kwargs["model"],
        )


def rubric_reply(score: int) -> str:
    crit = {c: {"reason": f"{c} ok", "score": score} for c in judge.CRITERIA}
    return json.dumps({**crit, "overall": score, "missing_parts": ["ears"], "extra_parts": [], "top_fixes": ["add ears"]})


def test_judge_rubric_scoring_and_request_shape():
    fake = FakeChat([rubric_reply(7)])
    cfg = EvalConfig(size=128, judge_model="test/vlm")
    r = judge.evaluate(make_pair(shape(), shape(), cfg), cfg, client=fake)
    assert r.score == pytest.approx((7 - 1) / 9)
    assert r.metrics["overall"] == 7 and r.details["top_fixes"] == ["add ears"]
    req = fake.requests[0]
    assert req["model"] == "test/vlm"
    assert req["response_format"]["json_schema"]["strict"] is True
    parts = req["messages"][1]["content"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 2 and images[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_judge_falls_back_when_schema_rejected_and_clamps_scores():
    reply = "```json\n" + rubric_reply(14) + "\n```"  # out-of-range score, fenced output
    fake = FakeChat([reply], reject_schema=True)
    r = judge.evaluate(make_pair(shape(), shape(), CFG), CFG, client=fake)
    assert r.score == pytest.approx(1.0)
    assert "response_format" not in fake.requests[-1]


def test_judge_pairwise_swaps_order():
    # A wins when shown first, and A wins again when shown second -> consistent preference for A.
    fake = FakeChat([json.dumps({"reason": "r", "winner": "first"}), json.dumps({"reason": "r", "winner": "second"})])
    res = judge.compare(shape(), shape(), shape("rectangle"), CFG, client=fake)
    assert res["p_a_better"] == 1.0 and res["winner"] == "a" and res["consistent"]
    # Position bias: "first" both times -> no real preference.
    fake = FakeChat([json.dumps({"reason": "r", "winner": "first"})] * 2)
    res = judge.compare(shape(), shape(), shape("rectangle"), CFG, client=fake)
    assert res["p_a_better"] == 0.5 and not res["consistent"]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def test_composite_renormalises_over_available_scores():
    results = {
        "a": EvalResult("a", 1.0),
        "b": EvalResult("b", 0.0),
        "c": EvalResult("c", None, skipped="no key"),
    }
    assert composite_score(results, {"a": 0.3, "b": 0.1, "c": 0.6}) == pytest.approx(0.75)
    assert composite_score({"c": results["c"]}, {"c": 1.0}) is None


def test_run_evals_subset_and_json_report(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(judge, "load_dotenv", lambda *a, **k: None)
    report = run_evals(shape(), shape(), ["silhouette", "edges", "color", "judge"], EvalConfig(size=128))
    d = json.loads(json.dumps(report.to_dict()))  # must be JSON-serialisable
    assert d["scores"]["silhouette"] == pytest.approx(1.0)
    assert d["results"]["judge"]["skipped"]
    assert d["composite"] == pytest.approx(1.0)  # judge skipped, the rest are perfect
    assert "composite" in report.summary()


def test_run_evals_rejects_unknown_names():
    with pytest.raises(ValueError):
        run_evals(shape(), shape(), ["nope"], CFG)


# --------------------------------------------------------------------------- #
# Model-backed evals (slow: downloads weights on first run)
# --------------------------------------------------------------------------- #


@slow
@pytest.mark.parametrize("name", ["pixel", "depth", "normals"])
def test_model_evals_identical_images_score_one(name):
    cfg = EvalConfig(size=256, normals_backend="depth")
    report = run_evals(shape(), shape(), [name], cfg)
    assert report.results[name].score == pytest.approx(1.0, abs=1e-3), report.results[name]


@slow
def test_model_evals_rank_similar_above_different():
    cfg = EvalConfig(size=256, normals_backend="depth")
    near = run_evals(shape(), shape(fill=(190, 70, 50)), ["pixel", "depth"], cfg)
    far = run_evals(shape(), shape("rectangle", fill=(40, 60, 200), stripe=False), ["pixel", "depth"], cfg)
    assert near.results["pixel"].score > far.results["pixel"].score


# --------------------------------------------------------------------------- #
# Project mode (the installed skill)
# --------------------------------------------------------------------------- #

FAST = ["silhouette", "edges", "color"]  # no model downloads, no network


def _save(img: Image.Image, path) -> str:
    img.save(path)
    return str(path)


def test_record_run_appends_history_and_replaces_latest(tmp_path):
    from render_eval.project import record_run

    ref = _save(shape(), tmp_path / "ref.png")
    first = _save(shape("rectangle"), tmp_path / "first.png")
    second = _save(shape(), tmp_path / "second.png")
    reports = tmp_path / "reports"

    r1 = record_run(ref, first, reports, EvalConfig(size=128), evals=FAST, label="v1")
    assert "first recorded run" in r1["critique"]
    assert "Critic: not run" in r1["critique"]
    (reports / "latest-run" / "stale.txt").write_text("from run 1")

    r2 = record_run(ref, second, reports, EvalConfig(size=128), evals=FAST, label="v2")
    history = json.loads((reports / "report.json").read_text())
    assert [r["label"] for r in history["runs"]] == ["v1", "v2"]
    assert history["runs"][1]["scores"]["silhouette"] == pytest.approx(1.0)
    assert "Since the previous run (first.png, v1)" in r2["critique"]
    assert "best run so far" in r2["critique"]

    latest = sorted(p.name for p in (reports / "latest-run").iterdir() if not p.name.startswith("."))
    assert "stale.txt" not in latest
    for name in ("00-aligned.png", "04-silhouette.png", "05-edges.png", "07-color.png", "overview.png",
                 "report.html", "critique.md", "run.json", "reference.png", "candidate.png"):
        assert name in latest
    assert (reports / "latest-run" / "critique.md").read_text() == r2["critique"]


def test_latest_run_refuses_to_delete_a_folder_it_did_not_create(tmp_path):
    from render_eval.project import record_run

    ref = _save(shape(), tmp_path / "ref.png")
    precious = tmp_path / "reports" / "latest-run" / "keep-me.txt"
    precious.parent.mkdir(parents=True)
    precious.write_text("not ours")
    with pytest.raises(RuntimeError):
        record_run(ref, ref, tmp_path / "reports", EvalConfig(size=128), evals=["silhouette"])
    assert precious.exists()
    assert not (tmp_path / "reports" / "report.json").exists()


# --------------------------------------------------------------------------- #
# Vectors
# --------------------------------------------------------------------------- #


def test_feature_names_cover_every_slot():
    from render_eval.vectorize import TOKEN_DIM, feature_names

    names = feature_names()
    assert len(names) == 8 * TOKEN_DIM == len(set(names))
    assert names[0] == "pixel.score" and names[-1] == "judge.present"
    assert "depth.nearer_r1c2" in names and "silhouette.extra_minus_missing_r0c3" in names


def test_encode_identical_pair_and_skipped_evals():
    from render_eval.vectorize import ORDER, encode

    report = run_evals(shape(), shape(), ["silhouette", "edges", "color"], EvalConfig(size=128))
    v = encode(report, embed_critique=False)
    assert v.tokens.shape == (8, 32)
    sil = v.tokens[ORDER.index("silhouette")]
    assert sil[0] == pytest.approx(1.0) and sil[-1] == 1.0
    assert np.allclose(sil[7:23], 0.0)  # no extra or missing area anywhere
    assert v.scores[ORDER.index("pixel")] is None
    assert np.all(v.tokens[ORDER.index("pixel")] == 0.0)  # skipped eval: all zeros, present flag 0


def test_silhouette_grid_localises_missing_area():
    from render_eval.vectorize import ORDER, encode

    full = shape(box=(40, 40, 280, 200), stripe=False)
    half = shape(box=(40, 40, 280, 200), stripe=False)
    ImageDraw.Draw(half).rectangle((160, 0, 320, 240), fill=(0, 0, 0, 0))  # render lacks the right half
    report = run_evals(full, half, ["silhouette"], EvalConfig(size=128, align="none"))
    grid = encode(report, embed_critique=False).tokens[ORDER.index("silhouette")][7:23].reshape(4, 4)
    assert grid[:, 2:].sum() < -0.5  # missing area shows up on the right
    assert abs(grid[:, :2].sum()) < 0.05  # left half matches


def test_embedding_and_judge_tokens_are_deterministic():
    from render_eval.vectorize import ORDER, encode

    class TextEmb:
        model = "fake/text"

        def embed_text(self, text):
            rng = np.random.default_rng(len(text))
            return rng.standard_normal(64).tolist()

    report = run_evals(shape(), shape(), ["silhouette"], EvalConfig(size=128))
    report.results["embedding"] = EvalResult("embedding", 0.9, {"cosine": 0.9, "model": "m", "dimensions": 3},
                                             {"reference_vector": [1.0, 0.0, 0.0], "render_vector": [0.9, 0.3, 0.0]})
    fake = FakeChat([rubric_reply(6)])
    report.results["judge"] = judge.evaluate(report.pair, CFG, client=fake)
    a = encode(report, embedder=TextEmb())
    b = encode(report, embedder=TextEmb())
    assert np.array_equal(a.tokens, b.tokens)
    j = a.tokens[ORDER.index("judge")]
    assert j[1] == pytest.approx(0.6) and j[9] == 1.0 and np.abs(j[10:26]).sum() > 0
    e = a.tokens[ORDER.index("embedding")]
    assert e[1] == 1.0 and np.abs(e[2:26]).sum() > 0


def test_record_run_stores_vectors_and_compares(tmp_path, capsys):
    from render_eval.cli import main
    from render_eval.project import record_run

    ref = _save(shape(), tmp_path / "ref.png")
    reports = tmp_path / "reports"
    for i, (kind, structure) in enumerate([("rectangle", "solo"), ("ellipse", "team"), ("ellipse", "team")]):
        cand = _save(shape(kind), tmp_path / f"c{i}.png")
        res = record_run(ref, cand, reports, EvalConfig(size=128), evals=FAST, label=f"r{i}",
                         meta={"structure": structure}, embed_critique=False)
    assert "Most similar earlier run by tokens" in res["critique"]
    assert (reports / "latest-run" / "vector.json").exists() and (reports / "latest-run" / "08-tokens.png").exists()
    history = json.loads((reports / "report.json").read_text())
    assert all(len(r["vector"]["tokens"]) == 8 for r in history["runs"])
    assert history["runs"][1]["meta"] == {"structure": "team"}

    assert main(["vectors", "--reports-dir", str(reports), "--export", str(tmp_path / "v.json")]) == 0
    out = capsys.readouterr().out
    assert "Most alike: B and C" in out
    exported = json.loads((tmp_path / "v.json").read_text())
    assert len(exported["feature_names"]) == 256 and len(exported["runs"][0]["vector"]) == 256
    assert main(["vectors", "--reports-dir", str(reports), "--delta", "--mode", "scores"]) == 0
