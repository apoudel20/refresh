"""Eval 6: semantic embedding similarity via OpenRouter.

Embeds both aligned images with a multimodal embedding model on OpenRouter and
takes the cosine similarity. The default model is ``voyageai/voyage-multimodal-3.5``
(Voyage AI is MongoDB's embedding team), so the same vectors can go straight into
a MongoDB Atlas Vector Search index. The HTTP client lives in
:mod:`render_eval.openrouter`.

Embedding similarity is tolerant of small pose and lighting changes, but coarse:
it says "same kind of thing", not "same shape". Cosine values between unrelated
photos are rarely near 0, so calibrate against baselines before trusting the
absolute number.

score = max(0, cosine)
"""

from __future__ import annotations

from typing import Any

from render_eval.base import EvalConfig, EvalResult, EvalSkipped, clip01
from render_eval.pair import ImagePair

NAME = "embedding"
DESCRIPTION = "Cosine similarity of OpenRouter multimodal image embeddings (Voyage by default)"


def evaluate(pair: ImagePair, cfg: EvalConfig, *, embedder: Any = None) -> EvalResult:
    from render_eval.openrouter import EmbeddingError, OpenRouterEmbedder, cosine_similarity

    if embedder is None:
        try:
            embedder = OpenRouterEmbedder(model=cfg.embedding_model, dimensions=cfg.embedding_dimensions)
        except EmbeddingError as exc:
            raise EvalSkipped(str(exc)) from exc

    res = embedder.embed_images([pair.pil("ref"), pair.pil("ren")])
    cos = cosine_similarity(res[0], res[1])
    metrics = {"cosine": round(cos, 6), "model": res.model, "dimensions": res.dimensions}
    details: dict[str, Any] = {"usage": res.usage}
    if cfg.keep_vectors:
        details["reference_vector"] = res[0]
        details["render_vector"] = res[1]
    return EvalResult(NAME, clip01(cos), metrics, details)
