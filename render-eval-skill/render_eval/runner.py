"""Run any subset of the eight evals on a reference/render pair and build one report."""

from __future__ import annotations

import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from render_eval import color, depth, edges, embedding, judge, normals, pixel, silhouette
from render_eval.base import EvalConfig, EvalResult, EvalSkipped
from render_eval.pair import ImageInput, ImagePair, make_pair

# Numbered as in the proposal: 1 pixel, 2 depth, 3 normals, 4 silhouette, 5 edges, 6 embedding, 7 color, 8 judge.
EVALS = {m.NAME: m for m in (pixel, depth, normals, silhouette, edges, embedding, color, judge)}
REMOTE = {"embedding", "judge"}  # network-bound; run in threads alongside the local models

# Starting weights for the composite. Uncalibrated: tune them against human ratings.
DEFAULT_WEIGHTS: dict[str, float] = {
    "silhouette": 0.20,
    "depth": 0.15,
    "normals": 0.15,
    "judge": 0.15,
    "pixel": 0.10,
    "edges": 0.10,
    "embedding": 0.10,
    "color": 0.05,
}


def jsonable(x: Any) -> Any:
    """Make numpy scalars JSON-safe and turn NaN/inf into None."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


@dataclass
class EvalReport:
    results: dict[str, EvalResult]
    composite: float | None
    weights: dict[str, float]
    pair: ImagePair = field(repr=False)
    config: EvalConfig = field(repr=False)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        cfg = asdict(self.config)
        cfg.pop("debug", None)
        return jsonable(
            {
                "composite": self.composite,
                "scores": {k: r.score for k, r in self.results.items()},
                "weights": self.weights,
                "results": {k: r.to_dict() for k, r in self.results.items()},
                "alignment": self.pair.meta,
                "config": cfg,
                "seconds": round(self.seconds, 3),
            }
        )

    def summary(self) -> str:
        lines = [f"{'eval':<11} {'score':>6}  key metrics", "-" * 72]
        for name, r in self.results.items():
            if r.ok:
                shown = ", ".join(f"{k}={_fmt(v)}" for k, v in list(r.metrics.items())[:4] if not isinstance(v, str))
                lines.append(f"{name:<11} {r.score:>6.3f}  {shown}")
            else:
                lines.append(f"{name:<11} {'-':>6}  {'skipped: ' + r.skipped if r.skipped else 'error: ' + str(r.error)}")
        lines.append("-" * 72)
        n_ok = sum(r.ok for r in self.results.values())
        comp = f"{self.composite:>6.3f}" if self.composite is not None else f"{'-':>6}"
        lines.append(f"{'composite':<11} {comp}  weighted mean over {n_ok}/{len(self.results)} evals")
        for w in self.pair.meta.get("warnings", []):
            lines.append(f"warning: {w}")
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def composite_score(results: dict[str, EvalResult], weights: dict[str, float]) -> float | None:
    """Weighted mean of the evals that produced a score; weights are renormalised over those."""
    pairs = [(weights.get(k, 0.0), r.score) for k, r in results.items() if r.ok and weights.get(k, 0.0) > 0]
    total = sum(w for w, _ in pairs)
    return None if total == 0 else float(sum(w * s for w, s in pairs) / total)


def _run_one(name: str, pair: ImagePair, cfg: EvalConfig) -> EvalResult:
    t0 = time.perf_counter()
    try:
        res = EVALS[name].evaluate(pair, cfg)
    except EvalSkipped as exc:
        res = EvalResult(name, None, skipped=str(exc))
    except Exception as exc:  # keep the other evals running; the traceback goes into details
        res = EvalResult(name, None, error=f"{type(exc).__name__}: {exc}", details={"traceback": traceback.format_exc()})
    res.seconds = time.perf_counter() - t0
    return res


def run_evals(
    reference: ImageInput,
    render: ImageInput,
    evals: list[str] | None = None,
    cfg: EvalConfig | None = None,
    *,
    reference_mask: ImageInput | None = None,
    render_mask: ImageInput | None = None,
    weights: dict[str, float] | None = None,
) -> EvalReport:
    """Score how closely ``render`` matches ``reference``. Returns per-eval results and a composite in [0, 1]."""
    cfg = cfg or EvalConfig()
    names = list(EVALS) if not evals else evals
    unknown = [n for n in names if n not in EVALS]
    if unknown:
        raise ValueError(f"unknown eval(s) {unknown}; choose from {list(EVALS)}")

    t0 = time.perf_counter()
    pair = make_pair(reference, render, cfg, reference_mask=reference_mask, render_mask=render_mask)
    results: dict[str, EvalResult] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        remote = {n: pool.submit(_run_one, n, pair, cfg) for n in names if n in REMOTE}
        for n in names:
            if n not in REMOTE:
                results[n] = _run_one(n, pair, cfg)
        for n, fut in remote.items():
            results[n] = fut.result()
    results = {n: results[n] for n in names}  # keep the requested order

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return EvalReport(
        results=results,
        composite=composite_score(results, w),
        weights={n: w.get(n, 0.0) for n in names},
        pair=pair,
        config=cfg,
        seconds=time.perf_counter() - t0,
    )
