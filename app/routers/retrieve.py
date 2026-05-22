import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.config import settings
from app.services import vector_store
from app.services.embeddings import embed_texts
from app.services.reranker import rerank

router = APIRouter()

LOG_PREFIX = "[retrieve]"


class RetrieveRequest(BaseModel):
    mode: Literal["chunks", "metadata"] = "chunks"
    query: str | None = None
    video_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=200)

    @model_validator(mode="after")
    def _check(self) -> "RetrieveRequest":
        if self.mode == "chunks" and not (self.query and self.query.strip()):
            raise ValueError("query is required when mode='chunks'")
        if self.mode == "metadata" and not self.video_ids:
            raise ValueError("video_ids is required when mode='metadata'")
        return self


class RetrievedChunk(BaseModel):
    rerank_score: float              # cross-encoder rerank score (final ranking signal)
    rrf_score: float | None = None   # server-side Qdrant RRF fused score (pre-rerank)
    text: str
    metadata: dict[str, Any]


class VideoMetadataOut(BaseModel):
    video_id: str
    title: str | None = None
    channel: str | None = None
    duration: int | None = None
    upload_date: str | None = None
    view_count: int | None = None
    like_count: int | None = None


class RetrieveResponse(BaseModel):
    mode: str
    query: str | None = None
    results: list[RetrievedChunk] = []
    metadata: list[VideoMetadataOut] = []


def _metadata_mode(req: RetrieveRequest) -> RetrieveResponse:
    print(f"{LOG_PREFIX} -> metadata mode video_ids={req.video_ids}")
    items: list[VideoMetadataOut] = []
    for vid in req.video_ids or []:
        md = vector_store.get_video_metadata(vid)
        if md is None:
            print(f"{LOG_PREFIX}    miss video_id={vid}")
            continue
        items.append(VideoMetadataOut(**md))
        print(
            f"{LOG_PREFIX}    hit  video_id={vid} title={md.get('title')!r} "
            f"views={md.get('view_count')} likes={md.get('like_count')}"
        )
    print(f"{LOG_PREFIX} <- metadata mode resolved={len(items)}/{len(req.video_ids or [])}")
    return RetrieveResponse(mode="metadata", metadata=items)


def _chunks_mode(req: RetrieveRequest) -> RetrieveResponse:
    t0 = time.perf_counter()
    candidate_k = req.candidate_k or settings.RERANK_CANDIDATE_K
    top_k = req.top_k or settings.RERANK_TOP_K
    query = req.query.strip()
    preview = query if len(query) <= 120 else query[:117] + "..."
    print(
        f"{LOG_PREFIX} -> chunks mode candidate_k={candidate_k} top_k={top_k} "
        f"video_ids={req.video_ids} query={preview!r}"
    )

    # 1. Encode dense query
    t_emb = time.perf_counter()
    query_vector = embed_texts([query])[0].tolist()
    print(
        f"{LOG_PREFIX}    embedded query dim={len(query_vector)} "
        f"({time.perf_counter() - t_emb:.3f}s)"
    )

    # 2. Hybrid (dense + sparse BM25) with native Qdrant RRF
    t_h = time.perf_counter()
    fused = vector_store.hybrid_search(
        query_text=query,
        query_vector=query_vector,
        candidate_k=candidate_k,
        video_ids=req.video_ids,
    )
    print(
        f"{LOG_PREFIX}    hybrid+rrf candidates={len(fused)} "
        f"({time.perf_counter() - t_h:.3f}s)"
    )

    if not fused:
        print(f"{LOG_PREFIX} <- chunks mode empty total={time.perf_counter() - t0:.3f}s")
        return RetrieveResponse(mode="chunks", query=query, results=[])

    # 3. Cross-encoder rerank
    rerank_input = [
        {"text": c["payload"].get("text", ""), "payload": c["payload"], "rrf_score": c["score"]}
        for c in fused
    ]

    t_r = time.perf_counter()
    reranked = rerank(query, rerank_input)
    print(
        f"{LOG_PREFIX}    reranked candidates={len(reranked)} model={settings.RERANKER_MODEL} "
        f"({time.perf_counter() - t_r:.3f}s)"
    )

    # 4. Top-K
    results: list[RetrievedChunk] = []
    for i, c in enumerate(reranked[:top_k], start=1):
        payload = dict(c["payload"])
        text = payload.pop("text", "")
        results.append(
            RetrievedChunk(
                rerank_score=c["rerank_score"],
                rrf_score=c["rrf_score"],
                text=text,
                metadata=payload,
            )
        )
        print(
            f"{LOG_PREFIX}      #{i} rerank={c['rerank_score']:.4f} "
            f"rrf={c['rrf_score']:.4f} "
            f"video_id={payload.get('video_id')} "
            f"[{payload.get('start_time')}-{payload.get('end_time')}]"
        )

    print(
        f"{LOG_PREFIX} <- chunks mode results={len(results)} "
        f"total={time.perf_counter() - t0:.3f}s"
    )
    return RetrieveResponse(mode="chunks", query=query, results=results)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    try:
        if req.mode == "metadata":
            return _metadata_mode(req)
        return _chunks_mode(req)
    except HTTPException:
        raise
    except Exception as e:
        print(f"{LOG_PREFIX} !! unhandled error: {e}")
        raise HTTPException(status_code=500, detail=f"retrieve failed: {e}")
