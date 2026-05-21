import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter

from app.schemas import IngestItemResult, IngestRequest, IngestResponse
from app.services import vector_store
from app.services.chunking import transcript_to_chunks
from app.services.embeddings import embed_texts, embedding_dimension
from app.services.youtube import extract_transcript, extract_video_metadata

router = APIRouter()

LOG_PREFIX = "[ingest]"
MAX_PARALLEL = 2


def _ingest_one(url: str) -> IngestItemResult:
    t0 = time.perf_counter()
    print(f"{LOG_PREFIX} -> request url={url}")

    try:
        meta = extract_video_metadata(url)
    except Exception as e:
        print(f"{LOG_PREFIX} !! metadata extraction failed url={url}: {e}")
        return IngestItemResult(url=url, success=False, error=f"metadata extraction failed: {e}")

    video_id = meta.get("video_id")
    if not video_id:
        print(f"{LOG_PREFIX} !! could not resolve video_id url={url}")
        return IngestItemResult(url=url, success=False, error="could not resolve video_id")
    print(
        f"{LOG_PREFIX}    metadata ok video_id={video_id} "
        f"title={meta.get('title')!r} channel={meta.get('channel')!r} "
        f"duration={meta.get('duration')}"
    )

    vector_store.ensure_collection(embedding_dimension())

    if vector_store.video_exists(video_id):
        print(
            f"{LOG_PREFIX} == already indexed video_id={video_id} "
            f"({time.perf_counter() - t0:.2f}s)"
        )
        return IngestItemResult(
            url=url,
            success=True,
            video_id=video_id,
            title=meta.get("title"),
            channel=meta.get("channel"),
            chunks_indexed=0,
            already_indexed=True,
        )

    try:
        t_tx = time.perf_counter()
        transcript = extract_transcript(video_id)
        print(
            f"{LOG_PREFIX}    transcript ok video_id={video_id} segments={len(transcript)} "
            f"({time.perf_counter() - t_tx:.2f}s)"
        )
    except Exception as e:
        print(f"{LOG_PREFIX} !! transcript extraction failed video_id={video_id}: {e}")
        return IngestItemResult(
            url=url,
            success=False,
            video_id=video_id,
            title=meta.get("title"),
            channel=meta.get("channel"),
            error=f"transcript extraction failed: {e}",
        )

    t_chunk = time.perf_counter()
    chunks = transcript_to_chunks(transcript)
    if not chunks:
        print(f"{LOG_PREFIX} !! no chunks produced video_id={video_id}")
        return IngestItemResult(
            url=url,
            success=False,
            video_id=video_id,
            title=meta.get("title"),
            channel=meta.get("channel"),
            error="no chunks produced from transcript",
        )
    print(
        f"{LOG_PREFIX}    chunked video_id={video_id} count={len(chunks)} "
        f"({time.perf_counter() - t_chunk:.2f}s)"
    )

    texts = [c["text"] for c in chunks]
    t_emb = time.perf_counter()
    embeddings = embed_texts(texts)
    print(
        f"{LOG_PREFIX}    embedded video_id={video_id} shape={embeddings.shape} "
        f"({time.perf_counter() - t_emb:.2f}s)"
    )

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
    print(
        f"{LOG_PREFIX}    upserted video_id={video_id} points={count} "
        f"({time.perf_counter() - t_up:.2f}s)"
    )

    print(
        f"{LOG_PREFIX} <- done video_id={video_id} chunks_indexed={count} "
        f"total={time.perf_counter() - t0:.2f}s"
    )

    return IngestItemResult(
        url=url,
        success=True,
        video_id=video_id,
        title=meta.get("title"),
        channel=meta.get("channel"),
        chunks_indexed=count,
        already_indexed=False,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    urls = [str(u) for u in req.urls]
    t_batch = time.perf_counter()
    print(f"{LOG_PREFIX} ==> batch start size={len(urls)} parallelism={MAX_PARALLEL}")

    results: list[IngestItemResult | None] = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        future_to_idx = {executor.submit(_ingest_one, url): i for i, url in enumerate(urls)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"{LOG_PREFIX} !! worker crashed url={urls[idx]}: {e}")
                results[idx] = IngestItemResult(url=urls[idx], success=False, error=str(e))

    ok = sum(1 for r in results if r and r.success)
    print(
        f"{LOG_PREFIX} <== batch done size={len(urls)} success={ok} "
        f"failed={len(urls) - ok} total={time.perf_counter() - t_batch:.2f}s"
    )

    return IngestResponse(results=[r for r in results if r is not None])
