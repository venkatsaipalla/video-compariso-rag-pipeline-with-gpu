import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import settings
from app.services.embeddings import embed_texts
from app.services.vector_store import get_client

router = APIRouter()

LOG_PREFIX = "[retrieve]"


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=50)


class RetrievedChunk(BaseModel):
    score: float
    text: str
    metadata: dict[str, Any]


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    t0 = time.perf_counter()
    preview = req.query if len(req.query) <= 120 else req.query[:117] + "..."
    print(f"{LOG_PREFIX} -> request top_k={req.top_k} query={preview!r}")

    t_emb = time.perf_counter()
    query_vector = embed_texts([req.query])[0].tolist()
    print(
        f"{LOG_PREFIX}    embedded query dim={len(query_vector)} "
        f"({time.perf_counter() - t_emb:.3f}s)"
    )

    t_q = time.perf_counter()
    hits = get_client().query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=req.top_k,
        with_payload=True,
    ).points
    print(
        f"{LOG_PREFIX}    qdrant query hits={len(hits)} "
        f"({time.perf_counter() - t_q:.3f}s)"
    )

    results = []
    for i, h in enumerate(hits, start=1):
        payload = dict(h.payload or {})
        text = payload.pop("text", "")
        results.append(RetrievedChunk(score=h.score, text=text, metadata=payload))
        print(
            f"{LOG_PREFIX}      #{i} score={h.score:.4f} "
            f"video_id={payload.get('video_id')} "
            f"[{payload.get('start_time')}-{payload.get('end_time')}]"
        )

    print(
        f"{LOG_PREFIX} <- done results={len(results)} "
        f"total={time.perf_counter() - t0:.3f}s"
    )

    return RetrieveResponse(query=req.query, results=results)
