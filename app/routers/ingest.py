import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter, Depends

from app.schemas import IngestItemResult, IngestRequest, IngestResponse
from app.security import require_api_key
from app.services import vector_store
from app.services.chunking import transcript_to_chunks
from app.services.embeddings import embed_texts, embedding_dimension
from app.services.logger import bind_context, get_logger
from app.services.youtube import extract_transcript, extract_video_metadata

router = APIRouter()

log = get_logger("ingest")
MAX_PARALLEL = 2


def _ingest_one(url: str, user_id: str) -> IngestItemResult:
    # Worker runs in a ThreadPoolExecutor thread — rebind context so log
    # lines from this thread carry the same user_id as the caller.
    bind_context(user_id=user_id)
    t0 = time.perf_counter()
    log.info(f"-> request url={url}")

    try:
        meta = extract_video_metadata(url)
    except Exception as e:
        log.error(f"metadata extraction failed url={url}: {e}")
        return IngestItemResult(url=url, success=False, error=f"metadata extraction failed: {e}")

    video_id = meta.get("video_id")
    if not video_id:
        log.error(f"could not resolve video_id url={url}")
        return IngestItemResult(url=url, success=False, error="could not resolve video_id")
    log.info(f"   metadata ok video_id={video_id} title={meta.get('title')!r} channel={meta.get('channel')!r} duration={meta.get('duration')}")

    vector_store.ensure_collection(embedding_dimension())

    if vector_store.video_exists(video_id):
        log.info(f"== already indexed video_id={video_id} ({time.perf_counter() - t0:.2f}s)")
        return IngestItemResult(url=url, success=True, video_id=video_id, chunks_indexed=0, already_indexed=True, metadata=meta)

    try:
        t_tx = time.perf_counter()
        transcript = extract_transcript(video_id)
        log.info(f"   transcript ok video_id={video_id} segments={len(transcript)} ({time.perf_counter() - t_tx:.2f}s)")
    except Exception as e:
        log.error(f"transcript extraction failed video_id={video_id}: {e}")
        return IngestItemResult(url=url, success=False, video_id=video_id, metadata=meta, error=f"transcript extraction failed: {e}")

    t_chunk = time.perf_counter()
    chunks = transcript_to_chunks(transcript)
    if not chunks:
        log.error(f"no chunks produced video_id={video_id}")
        return IngestItemResult(url=url, success=False, video_id=video_id, metadata=meta, error="no chunks produced from transcript")
    log.info(f"   chunked video_id={video_id} count={len(chunks)} ({time.perf_counter() - t_chunk:.2f}s)")

    texts = [c["text"] for c in chunks]
    t_emb = time.perf_counter()
    embeddings = embed_texts(texts)
    log.info(f"   embedded video_id={video_id} shape={tuple(embeddings.shape)} ({time.perf_counter() - t_emb:.2f}s)")

    payload_meta = {
        "video_id": video_id,
        "title": meta.get("title"),
        "channel": meta.get("channel"),
        "duration": meta.get("duration"),
        "upload_date": meta.get("upload_date"),
        "view_count": meta.get("view_count"),
        "like_count": meta.get("like_count"),
    }

    t_up = time.perf_counter()
    count = vector_store.upsert_chunks(chunks, embeddings, payload_meta)
    log.info(f"   upserted (dense+sparse) video_id={video_id} points={count} ({time.perf_counter() - t_up:.2f}s)")

    log.info(f"<- done video_id={video_id} chunks_indexed={count} total={time.perf_counter() - t0:.2f}s")

    return IngestItemResult(url=url, success=True, video_id=video_id, chunks_indexed=count, already_indexed=False, metadata=meta)


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest(req: IngestRequest) -> IngestResponse:
    user_id = req.user_id or ""
    bind_context(user_id=user_id)

    urls = [str(u) for u in req.urls]
    t_batch = time.perf_counter()
    log.info(f"==> batch start size={len(urls)} parallelism={MAX_PARALLEL}")

    results: list[IngestItemResult | None] = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        future_to_idx = {executor.submit(_ingest_one, url, user_id): i for i, url in enumerate(urls)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                log.exception(f"worker crashed url={urls[idx]}: {e}")
                results[idx] = IngestItemResult(url=urls[idx], success=False, error=str(e))

    ok = sum(1 for r in results if r and r.success)
    log.info(f"<== batch done size={len(urls)} success={ok} failed={len(urls) - ok} total={time.perf_counter() - t_batch:.2f}s")

    return IngestResponse(results=[r for r in results if r is not None])
