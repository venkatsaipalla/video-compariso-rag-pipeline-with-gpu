from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers.ingest import router as ingest_router
from app.routers.retrieve import router as retrieve_router
from app.services.embeddings import get_device, get_embedding_model
from app.services.logger import get_logger
from app.services.reranker import get_reranker
from app.services.sparse import get_sparse_model
from app.services.vector_store import ensure_collection

load_dotenv()

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"[{settings.APP_NAME}] device: {get_device()}")
    model = get_embedding_model()
    dim = model.get_embedding_dimension()
    log.info(f"[{settings.APP_NAME}] embedding model loaded (dim={dim})")
    ensure_collection(dim)
    log.info(f"[{settings.APP_NAME}] qdrant collection ready: {settings.QDRANT_COLLECTION}")
    # Warm the FastEmbed sparse BM25 model + cross-encoder so the first
    # request isn't blocked on downloads / NLTK assets.
    get_sparse_model()
    log.info(f"[{settings.APP_NAME}] sparse bm25 model loaded (Qdrant/bm25, lemmatized)")
    get_reranker()
    log.info(f"[{settings.APP_NAME}] reranker loaded: {settings.RERANKER_MODEL}")

    if settings.RETRIEVAL_API_KEY:
        log.info(f"[{settings.APP_NAME}] API key auth ENABLED (header: X-API-Key)")
    else:
        log.warning(f"[{settings.APP_NAME}] RETRIEVAL_API_KEY unset — auth DISABLED")
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(retrieve_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.reload,
    )
