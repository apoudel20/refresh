"""render-eval: grade how closely a render of a 3D model matches a reference image.

Eight image-to-image evals, each returning a score in [0, 1] (higher = closer match)
plus raw metrics:

1. ``pixel``       PSNR / SSIM / LPIPS
2. ``depth``       Depth Anything V2 depth maps, scale/shift-invariant comparison
3. ``normals``     Marigold (or depth-derived) surface normals, angular error
4. ``silhouette``  foreground mask IoU / Dice / outline distance
5. ``edges``       Canny edge F-score and Chamfer distance
6. ``embedding``   OpenRouter multimodal embeddings (Voyage), cosine similarity
7. ``color``       CIELAB palette earth mover's distance, Delta E
8. ``judge``       vision LLM on OpenRouter with a rubric (plus pairwise ``compare``)

Quick start::

    from render_eval import run_evals, record_run, encode
    report = run_evals("reference.png", "render.png")
    print(report.summary())

    # project mode: history file, latest-run folder, critique text, vectors
    res = record_run("reference.png", "render.png", "reports", label="round 3")
    print(res["critique"])
"""

from render_eval.base import EvalConfig, EvalResult, EvalSkipped
from render_eval.judge import compare
from render_eval.pair import ImagePair, make_pair
from render_eval.project import record_run
from render_eval.runner import DEFAULT_WEIGHTS, EVALS, EvalReport, composite_score, run_evals
from render_eval.vectorize import RunVector, encode, feature_names

__all__ = [
    "DEFAULT_WEIGHTS",
    "EVALS",
    "EvalConfig",
    "EvalReport",
    "EvalResult",
    "EvalSkipped",
    "ImagePair",
    "RunVector",
    "compare",
    "composite_score",
    "encode",
    "feature_names",
    "make_pair",
    "record_run",
    "run_evals",
]
