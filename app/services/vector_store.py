from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import settings
from app.services.sparse import encode_documents, encode_query

_client: QdrantClient | None = None

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

_METADATA_FIELDS = (
    "video_id",
    "title",
    "channel",
    "duration",
    "upload_date",
    "view_count",
    "like_count",
)


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if settings.QDRANT_URL:
            _client = QdrantClient(
                url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
            )
        else:
            _client = QdrantClient(path=settings.QDRANT_PATH)
    return _client


def _is_compatible(client: QdrantClient) -> bool:
    info = client.get_collection(settings.QDRANT_COLLECTION)
    params = info.config.params
    vectors = params.vectors
    sparse = params.sparse_vectors or {}
    return (
        isinstance(vectors, dict)
        and DENSE_VECTOR in vectors
        and SPARSE_VECTOR in sparse
    )


def ensure_collection(vector_size: int) -> None:
    client = get_client()
    if client.collection_exists(settings.QDRANT_COLLECTION):
        if not _is_compatible(client):
            print(
                f"[vector_store] existing collection {settings.QDRANT_COLLECTION!r} "
                "is incompatible with named dense+sparse layout — recreating"
            )
            client.delete_collection(settings.QDRANT_COLLECTION)

    if not client.collection_exists(settings.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )

    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="video_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _video_filter(video_ids: list[str] | None) -> models.Filter | None:
    if not video_ids:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="video_id", match=models.MatchAny(any=video_ids)
            )
        ]
    )


def video_exists(video_id: str) -> bool:
    client = get_client()
    if not client.collection_exists(settings.QDRANT_COLLECTION):
        return False
    hits, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id", match=models.MatchValue(value=video_id)
                )
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(hits) > 0


def upsert_chunks(
    chunks: list[dict[str, Any]],
    embeddings,
    metadata: dict[str, Any],
) -> int:
    client = get_client()
    texts = [c["text"] for c in chunks]
    sparse_vectors = encode_documents(texts)
    points: list[models.PointStruct] = []
    for chunk, embedding, sparse in zip(chunks, embeddings, sparse_vectors):
        payload = {
            "text": chunk["text"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            **metadata,
        }
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR: embedding.tolist(),
                    SPARSE_VECTOR: sparse,
                },
                payload=payload,
            )
        )
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    return len(points)


def hybrid_search(
    query_text: str,
    query_vector: list[float],
    candidate_k: int,
    video_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Dense + sparse(BM25) with server-side RRF fusion.

    Returns [{id, score, payload}, ...] sorted by fused score (best first).
    """
    flt = _video_filter(video_ids)
    sparse_query = encode_query(query_text)

    result = get_client().query_points(
        collection_name=settings.QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=query_vector,
                using=DENSE_VECTOR,
                limit=candidate_k,
                filter=flt,
            ),
            models.Prefetch(
                query=sparse_query,
                using=SPARSE_VECTOR,
                limit=candidate_k,
                filter=flt,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidate_k,
        with_payload=True,
        query_filter=flt,
    )

    return [
        {"id": str(p.id), "score": float(p.score), "payload": dict(p.payload or {})}
        for p in result.points
    ]


def get_video_metadata(video_id: str) -> dict[str, Any] | None:
    client = get_client()
    if not client.collection_exists(settings.QDRANT_COLLECTION):
        return None
    hits, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id", match=models.MatchValue(value=video_id)
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not hits:
        return None
    payload = hits[0].payload or {}
    return {k: payload.get(k) for k in _METADATA_FIELDS}
