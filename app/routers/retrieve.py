import time

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.schemas import (
    RetrievedChunk,
    RetrieveRequest,
    RetrieveResponse,
    VideoMetadata,
)
from app.security import require_api_key
from app.services import vector_store
from app.services.embeddings import embed_texts
from app.services.logger import bind_context, get_logger
from app.services.reranker import rerank

router = APIRouter()

log = get_logger("retrieve")


def _metadata_mode(req: RetrieveRequest) -> RetrieveResponse:
    log.info(f"-> metadata mode video_ids={req.video_ids}")
    items: list[VideoMetadata] = []
    for vid in req.video_ids or []:
        md = vector_store.get_video_metadata(vid)
        if md is None:
            log.info(f"   miss video_id={vid}")
            continue
        items.append(VideoMetadata(**md))
        log.info(f"   hit  video_id={vid} title={md.get('title')!r} views={md.get('view_count')} likes={md.get('like_count')}")
    log.info(f"<- metadata mode resolved={len(items)}/{len(req.video_ids or [])}")
    return RetrieveResponse(mode="metadata", metadata=items)


def _chunks_mode(req: RetrieveRequest) -> RetrieveResponse:
    t0 = time.perf_counter()
    candidate_k = req.candidate_k or settings.RERANK_CANDIDATE_K
    top_k = req.top_k or settings.RERANK_TOP_K
    query = req.query.strip()
    preview = query if len(query) <= 120 else query[:117] + "..."
    log.info(f"-> chunks mode candidate_k={candidate_k} top_k={top_k} video_ids={req.video_ids} query={preview!r}")

    # 1. Encode dense query
    t_emb = time.perf_counter()
    query_vector = embed_texts([query])[0].tolist()
    log.info(f"   embedded query dim={len(query_vector)} ({time.perf_counter() - t_emb:.3f}s)")

    # 2. Hybrid (dense + sparse BM25) with native Qdrant RRF
    t_h = time.perf_counter()
    fused = vector_store.hybrid_search(query_text=query, query_vector=query_vector, candidate_k=candidate_k, video_ids=req.video_ids)
    log.info(f"   hybrid+rrf candidates={len(fused)} ({time.perf_counter() - t_h:.3f}s)")

    if not fused:
        log.info(f"<- chunks mode empty total={time.perf_counter() - t0:.3f}s")
        return RetrieveResponse(mode="chunks", query=query, results=[])

    # 3. Cross-encoder rerank
    rerank_input = [{"text": c["payload"].get("text", ""), "payload": c["payload"], "rrf_score": c["score"]} for c in fused]

    t_r = time.perf_counter()
    reranked = rerank(query, rerank_input)
    log.info(f"   reranked candidates={len(reranked)} model={settings.RERANKER_MODEL} ({time.perf_counter() - t_r:.3f}s)")

    # 4. Top-K
    results: list[RetrievedChunk] = []
    for i, c in enumerate(reranked[:top_k], start=1):
        payload = dict(c["payload"])
        text = payload.pop("text", "")
        results.append(RetrievedChunk(rerank_score=c["rerank_score"], rrf_score=c["rrf_score"], text=text, metadata=payload))
        log.info(f"     #{i} rerank={c['rerank_score']:.4f} rrf={c['rrf_score']:.4f} video_id={payload.get('video_id')} [{payload.get('start_time')}-{payload.get('end_time')}]")

    log.info(f"<- chunks mode results={len(results)} total={time.perf_counter() - t0:.3f}s")
    return RetrieveResponse(mode="chunks", query=query, results=results)


@router.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_key)])
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    bind_context(user_id=req.user_id or "", session_id=req.session_id or "")
    try:
        if req.mode == "metadata":
            return _metadata_mode(req)
        return _chunks_mode(req)
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"unhandled error: {e}")
        raise HTTPException(status_code=500, detail=f"retrieve failed: {e}")
