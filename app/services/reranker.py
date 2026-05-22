from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.config import settings
from app.services.embeddings import get_device

_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(settings.RERANKER_MODEL, device=get_device())
    return _model


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Each candidate must have a "text" field. Returns sorted copy with rerank_score."""
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    scores = get_reranker().predict(pairs, show_progress_bar=False)
    enriched = []
    for c, s in zip(candidates, scores):
        item = dict(c)
        item["rerank_score"] = float(s)
        enriched.append(item)
    enriched.sort(key=lambda c: c["rerank_score"], reverse=True)
    return enriched
