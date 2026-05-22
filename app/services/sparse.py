"""Sparse BM25 vectors via FastEmbed (`Qdrant/bm25`).

FastEmbed's `Qdrant/bm25` is the standard sparse-BM25 model designed to pair
with Qdrant's `Modifier.IDF` sparse vector config. It ships with English
Snowball stemming by default. We disable that and plug in a WordNet
lemmatizer so query and document terms collapse to dictionary lemmas instead
of stems — this matches the request to prefer lemmatization where applicable.

NLTK's WordNet/punkt assets are downloaded on demand the first time the
module is imported.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import nltk
from fastembed import SparseTextEmbedding
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from qdrant_client import models

_NLTK_PKGS = ("wordnet", "omw-1.4", "averaged_perceptron_tagger_eng", "punkt_tab")


def _ensure_nltk() -> None:
    for pkg in _NLTK_PKGS:
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg in ("wordnet", "omw-1.4") else pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()
_lemmatizer = WordNetLemmatizer()
_model: SparseTextEmbedding | None = None
_lock = threading.Lock()


def get_sparse_model() -> SparseTextEmbedding:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                # disable_stemmer=True: we feed already-lemmatized tokens via
                # FastEmbed's text path, so its Snowball stemmer must be off.
                _model = SparseTextEmbedding(
                    model_name="Qdrant/bm25",
                    disable_stemmer=True,
                )
    return _model


_POS_MAP = {
    "J": wordnet.ADJ,
    "V": wordnet.VERB,
    "N": wordnet.NOUN,
    "R": wordnet.ADV,
}


def _wn_pos(tag: str) -> str:
    return _POS_MAP.get(tag[:1].upper(), wordnet.NOUN)


@lru_cache(maxsize=200_000)
def _lemma(token: str, pos: str) -> str:
    return _lemmatizer.lemmatize(token, pos)


def _lemmatize_text(text: str) -> str:
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return " ".join(_lemma(tok.lower(), _wn_pos(tag)) for tok, tag in tagged)


def _to_sparse(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


def encode_documents(texts: list[str]) -> list[models.SparseVector]:
    if not texts:
        return []
    lemmatized = [_lemmatize_text(t) for t in texts]
    return [_to_sparse(e) for e in get_sparse_model().embed(lemmatized)]


def encode_query(text: str) -> models.SparseVector:
    lemmatized = _lemmatize_text(text)
    return _to_sparse(next(get_sparse_model().query_embed(lemmatized)))
