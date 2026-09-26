"""CLI for the reference-vs-render evals.

    render-eval run reference.png render.png                 # all 8 evals, table output
    render-eval run ref.png render.png -e silhouette,depth   # a subset
    render-eval run ref.png render.png --json -o report.json --debug-dir out/
    render-eval compare ref.png render_a.png render_b.png    # pairwise VLM judge
    render-eval report ref.png a.png b.png -o outputs/report # step-by-step HTML + PNG + JSON
    render-eval critique ref.png render.png                  # project mode: reports/ + critique text
    render-eval vectors --reports-dir reports               # compare recorded runs as vectors
    render-eval list
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

from PIL import Image

from render_eval.base import EvalConfig, EvalSkipped
from render_eval.pair import to_uint8
from render_eval.runner import DEFAULT_WEIGHTS, EVALS, EvalReport, jsonable, run_evals


def _config_from_args(args: argparse.Namespace) -> EvalConfig:
    names = {f.name for f in fields(EvalConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in names and v is not None}
    return EvalConfig(**kwargs)


def _add_config_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("preprocessing")
    g.add_argument("--size", type=int, help="working resolution in px (default 512)")
    g.add_argument("--align", choices=["bbox", "none"], help="bbox: crop each image to its object (default)")
    g.add_argument("--background", choices=["neutral", "keep"], help="neutral: put both objects on grey (default)")
    g.add_argument("--mask-model", dest="mask_model", help="rembg model for images without alpha (default birefnet-general-lite)")
    g = p.add_argument_group("models")
    g.add_argument("--device", help="torch device (default: mps, cuda or cpu)")
    g.add_argument("--depth-model", dest="depth_model", help="Hugging Face depth model id")
    g.add_argument("--normals-backend", dest="normals_backend", choices=["marigold", "depth"])
    g.add_argument("--embedding-model", dest="embedding_model", help="OpenRouter embedding model id")
    g.add_argument("--embedding-dimensions", dest="embedding_dimensions", type=int)
    g.add_argument("--judge-model", dest="judge_model", help="OpenRouter vision model id for the judge")


def write_debug(report: EvalReport, out_dir: Path) -> list[Path]:
    from render_eval.report import alignment_panel

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    panels = {"alignment": alignment_panel(report.pair)}
    for r in report.results.values():
        panels.update(r.artifacts)
    for name, arr in panels.items():
        path = out_dir / f"{name}.png"
        Image.fromarray(to_uint8(arr)).save(path)
        written.append(path)
    return written


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    cfg.debug = bool(args.debug_dir)
    cfg.keep_vectors = args.vectors
    evals = [e.strip() for e in args.evals.split(",")] if args.evals else None
    weights = json.loads(args.weights) if args.weights else None
    report = run_evals(
        args.reference, args.render, evals, cfg,
        reference_mask=args.reference_mask, render_mask=args.render_mask, weights=weights,
    )
    data = report.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=2))
    if args.debug_dir:
        paths = write_debug(report, Path(args.debug_dir))
        print(f"wrote {len(paths)} debug image(s) to {args.debug_dir}", file=sys.stderr)
    print(json.dumps(data, indent=2) if args.json else report.summary())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from render_eval.judge import compare

    cfg = _config_from_args(args)
    try:
        res = compare(args.reference, args.render_a, args.render_b, cfg)
    except EvalSkipped as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(jsonable(res), indent=2))
    else:
        print(f"P(A closer to reference) = {res['p_a_better']:.2f}  ->  winner: {res['winner']}"
              f"  ({'consistent' if res['consistent'] else 'order-dependent'} across both orderings)")
        for v in res["verdicts"]:
            print(f"- {v['order']}: {v['winner']}: {v['reason']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from render_eval.report import build_report, summary_table

    cfg = _config_from_args(args)
    evals = [e.strip() for e in args.evals.split(",")] if args.evals else None
    weights = json.loads(args.weights) if args.weights else None
    res = build_report(args.reference, args.candidates, args.out, cfg, evals=evals, weights=weights)
    print(summary_table(res["runs"]))
    print("\n".join(f"wrote {p}" for p in res["files"]), file=sys.stderr)
    return 0


def _quiet_models() -> None:
    """Keep stdout/stderr readable for agents: no download bars or library deprecation chatter."""
    import os
    import warnings

    for k, v in {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_VERBOSITY": "error", "TRANSFORMERS_VERBOSITY": "error",
                 "DIFFUSERS_VERBOSITY": "error", "TQDM_DISABLE": "1"}.items():
        os.environ.setdefault(k, v)
    warnings.filterwarnings("ignore")


def cmd_critique(args: argparse.Namespace) -> int:
    from render_eval.project import record_run

    _quiet_models()
    cfg = _config_from_args(args)
    evals = [e.strip() for e in args.evals.split(",")] if args.evals else None
    weights = json.loads(args.weights) if args.weights else None
    meta = json.loads(args.meta) if args.meta else None
    res = record_run(args.reference, args.render, args.reports_dir, cfg, evals=evals, weights=weights,
                     label=args.label, meta=meta)
    if args.json:
        print(json.dumps(jsonable(res["entry"]), indent=2))
    else:
        sys.stdout.write(res["critique"])
    return 0


def cmd_vectors(args: argparse.Namespace) -> int:
    import numpy as np

    from render_eval.project import REPORT_FILE, load_history
    from render_eval.vectorize import ORDER, distance_matrix, feature_names, history_vectors, token_distances

    history = load_history(Path(args.reports_dir) / REPORT_FILE)
    rows = history_vectors(history["runs"], delta=args.delta)
    if args.reference:
        rows = [(r, v) for r, v in rows if Path(r.get("reference") or "").name == Path(args.reference).name]
    if args.last:
        rows = rows[-args.last:]
    skipped = sum("vector" not in r for r in history["runs"])
    if not rows:
        print("no runs with vectors" + (" (need two runs per reference for --delta)" if args.delta else ""), file=sys.stderr)
        return 1

    mats = [v.tokens if args.mode == "tokens" else v.score_array() for _, v in rows]
    keys = [chr(65 + i) if i < 26 else f"R{i}" for i in range(len(rows))]
    kind = "change since the previous run" if args.delta else "run"
    dims = "8 x 32 = 256 numbers" if args.mode == "tokens" else "8 numbers"
    print(f"{len(rows)} runs, one vector per {kind} ({args.mode}: {dims})"
          + (f"; {skipped} older runs have no vector" if skipped else ""))
    print(f"{'key':<4} {'run':<30} {'label / structure':<34} score vector ({' '.join(n[:4] for n in ORDER)})")
    for k, (run, v) in zip(keys, rows):
        tag = run.get("label") or ""
        if run.get("meta", {}).get("structure"):
            tag = f"{tag} [{run['meta']['structure']}]".strip()
        vals = " ".join("  n/a" if s is None else f"{s:+.2f}" if args.delta else f"{s:5.2f}" for s in v.scores)
        print(f"{k:<4} {run['id'][:30]:<30} {tag[:34]:<34} {vals}")

    if len(rows) > 1:
        dm = distance_matrix(mats)
        print(f"\nEuclidean distance between {'changes' if args.delta else 'runs'} ({args.mode}); smaller = more alike")
        print("     " + "".join(f"{k:>7}" for k in keys))
        for k, row in zip(keys, dm):
            print(f"{k:<5}" + "".join(f"{x:7.2f}" for x in row))
        iu = np.triu_indices(len(rows), 1)
        order = np.argsort(dm[iu])
        picks = [("Difference", order[0])] if len(order) == 1 else [("Most alike", order[0]), ("Most different", order[-1])]
        for title, idx in picks:
            i, j = iu[0][idx], iu[1][idx]
            if args.mode == "tokens":
                per, what = token_distances(rows[i][1].tokens, rows[j][1].tokens), "Tokens"
            else:
                per = {n: abs(float(a) - float(b)) for n, a, b in zip(ORDER, mats[i], mats[j])}
                what = "Step scores"
            top = ", ".join(f"{n} {d:.2f}" for n, d in sorted(per.items(), key=lambda kv: -kv[1])[:3])
            print(f"{title}: {keys[i]} and {keys[j]}, distance {dm[i, j]:.2f}. {what} that differ most: {top}")

    if args.export:
        out = {
            "mode": args.mode,
            "delta": args.delta,
            "feature_names": feature_names() if args.mode == "tokens" else list(ORDER),
            "runs": [
                {"key": k, "id": r["id"], "label": r.get("label"), "meta": r.get("meta", {}),
                 "composite": r.get("composite"), "candidate": r.get("candidate"),
                 "vector": [round(float(x), 5) for x in np.asarray(m).reshape(-1)]}
                for k, (r, _), m in zip(keys, rows, mats)
            ],
        }
        Path(args.export).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.export}", file=sys.stderr)
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    for i, (name, mod) in enumerate(EVALS.items(), 1):
        print(f"{i}. {name:<11} weight {DEFAULT_WEIGHTS.get(name, 0):.2f}  {mod.DESCRIPTION}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="render-eval", description="Grade how closely a render of a 3D model matches a reference image")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="Run evals on a reference/render pair")
    pr.add_argument("reference", help="reference image (photo or render of the target object)")
    pr.add_argument("render", help="render of the Blender model (transparent background works best)")
    pr.add_argument("--evals", "-e", help=f"comma-separated subset of: {','.join(EVALS)} (default: all)")
    pr.add_argument("--reference-mask", dest="reference_mask", help="optional foreground mask for the reference")
    pr.add_argument("--render-mask", dest="render_mask", help="optional foreground mask for the render")
    pr.add_argument("--weights", help='JSON overrides for composite weights, e.g. \'{"judge": 0.3}\'')
    pr.add_argument("--json", action="store_true", help="print the full JSON report instead of a table")
    pr.add_argument("--out", "-o", help="also write the JSON report to this file")
    pr.add_argument("--debug-dir", dest="debug_dir", help="write visual comparison PNGs here")
    pr.add_argument("--vectors", action="store_true", help="include embedding vectors in the JSON report")
    _add_config_args(pr)
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("compare", help="Pairwise VLM judge: which of two renders is closer to the reference?")
    pc.add_argument("reference")
    pc.add_argument("render_a")
    pc.add_argument("render_b")
    pc.add_argument("--json", action="store_true")
    _add_config_args(pc)
    pc.set_defaults(func=cmd_compare)

    pp = sub.add_parser("report", help="Step-by-step HTML/PNG/JSON report: one reference vs many candidates")
    pp.add_argument("reference")
    pp.add_argument("candidates", nargs="+", help="candidate images (renders) to compare against the reference")
    pp.add_argument("--out", "-o", default="outputs/report", help="output folder (default outputs/report)")
    pp.add_argument("--evals", "-e", help=f"comma-separated subset of: {','.join(EVALS)} (default: all)")
    pp.add_argument("--weights", help="JSON overrides for composite weights")
    _add_config_args(pp)
    pp.set_defaults(func=cmd_report)

    pk = sub.add_parser(
        "critique",
        help="Project mode: score a render, append to reports/report.json, rewrite reports/latest-run/, print the critique",
    )
    pk.add_argument("reference")
    pk.add_argument("render")
    pk.add_argument("--reports-dir", dest="reports_dir", default="reports", help="reports folder (default ./reports)")
    pk.add_argument("--label", help="short note stored with the run, e.g. what changed in the model")
    pk.add_argument("--evals", "-e", help=f"comma-separated subset of: {','.join(EVALS)} (default: all)")
    pk.add_argument("--weights", help="JSON overrides for composite weights")
    pk.add_argument("--meta", help='JSON stored with the run, e.g. \'{"structure": "geometry+fur+materials"}\'')
    pk.add_argument("--json", action="store_true", help="print the run record as JSON instead of the critique")
    _add_config_args(pk)
    pk.set_defaults(func=cmd_critique)

    pv = sub.add_parser("vectors", help="Compare recorded runs by their score vectors or token matrices")
    pv.add_argument("--reports-dir", dest="reports_dir", default="reports", help="reports folder (default ./reports)")
    pv.add_argument("--mode", choices=["tokens", "scores"], default="tokens",
                    help="tokens: 8 x 32 token matrix (default); scores: the 8 step scores")
    pv.add_argument("--delta", action="store_true", help="compare what each run changed vs the previous run")
    pv.add_argument("--reference", help="only runs against this reference (matched by file name)")
    pv.add_argument("--last", type=int, help="only the last N runs")
    pv.add_argument("--export", help="write the vectors and feature names to this JSON file")
    pv.set_defaults(func=cmd_vectors)

    pl = sub.add_parser("list", help="List the evals and their composite weights")
    pl.set_defaults(func=cmd_list)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
