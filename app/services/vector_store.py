from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config import settings

_client: QdrantClient | None = None


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


def ensure_collection(vector_size: int) -> None:
    client = get_client()
    if not client.collection_exists(settings.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="video_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def video_exists(video_id: str) -> bool:
    client = get_client()
    if not client.collection_exists(settings.QDRANT_COLLECTION):
        return False
    hits, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
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
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "text": chunk["text"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                **metadata,
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    return len(points)
