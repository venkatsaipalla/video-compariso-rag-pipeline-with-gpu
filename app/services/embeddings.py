from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None
_device: str | None = None


def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{settings.APP_NAME}] device: {_device}")
    return _device


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL, device=get_device())
    return _model


def embed_texts(texts: list[str]):
    model = get_embedding_model()
    return model.encode(
        texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def embedding_dimension() -> int:
    return get_embedding_model().get_embedding_dimension()
