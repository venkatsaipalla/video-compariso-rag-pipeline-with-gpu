# Video Comparison RAG — GPU Engine

GPU-backed **ingestion + retrieval engine** for a multi-agentic YouTube video analysis system.

This repo is the **retrieval + inference layer**. It is paired with a separate
**cognitive layer** repo built on the **Google ADK** framework (orchestration, memory,
multi-agent reasoning, response synthesis). That repo calls this service's HTTP
endpoints as tools — this repo owns every GPU-heavy operation (embeddings, sparse
encoding, cross-encoder reranking) and the vector database.

> ⚠️ **Requires a CUDA-capable GPU.** Models load onto `cuda` when available and fall
> back to CPU only for local dev. Production is expected to run on GPU.

---

## What it does

- **Ingestion** — YouTube URL → metadata + transcript → windowing → recursive chunking
  (overlap) → dense + sparse embeddings → Qdrant.
- **Retrieval** — hybrid search + fusion + reranking over the indexed transcripts.

It does **not** do reasoning or answer synthesis — that's the ADK repo's job.

---

## Endpoints

All protected endpoints require the header `X-API-Key: <RETRIEVAL_API_KEY>`.

| Method | Path        | Auth | Purpose |
|--------|-------------|------|---------|
| GET    | `/health`   | no   | Liveness check |
| POST   | `/ingest`   | yes  | Ingest a list of YouTube URLs (2 processed in parallel) |
| POST   | `/retrieve` | yes  | Retrieve in `metadata` or `chunks` mode |

### `/retrieve` — two modes

The agent picks a mode based on the question:

- **`metadata`** — returns only stored video metadata (title, channel, views, likes,
  duration, upload date) for the given `video_ids`. Used for performance/virality
  questions like *"Which video performed better?"* — these need **stats, not transcript
  chunks**, so we skip retrieval entirely (cheap + accurate).
- **`chunks`** — semantic transcript retrieval for a natural-language `query`,
  optionally scoped to specific `video_ids`.

#### Why metadata-first
Comparison/performance queries are answered from numbers (views, likes, duration),
not text. Returning metadata directly avoids needless embedding + reranking cost and
prevents the LLM from hallucinating stats out of transcript text.

---

## Retrieval strategy (`chunks` mode)

```
Query
 ├── Dense retriever  (vector / semantic)
 ├── Sparse retriever (BM25, IDF)
 ↓
RRF fusion          (Reciprocal Rank Fusion, server-side in Qdrant)
 ↓
Cross-Encoder rerank
 ↓
Top-K results
```

1. **Dense** — top-15 candidates via cosine similarity.
2. **Sparse (BM25)** — top-15 via Qdrant native sparse vectors (`Modifier.IDF`).
3. **RRF fusion** — both lists fused server-side in a single Qdrant query.
4. **Cross-encoder rerank** — the 15 fused candidates re-scored by a cross-encoder.
5. **Top-K** — best 3 returned (`candidate_k` / `top_k` are tunable per request).

Each result carries `rerank_score` (final ranking signal), `rrf_score`, the chunk
`text`, and metadata (`video_id`, `title`, `channel`, `start_time`, `end_time`, …) for
timestamp-accurate references.

---

## Models

| Role | Model | Notes |
|------|-------|-------|
| Dense embeddings | `BAAI/bge-large-en-v1.5` | 1024-dim, runs on CUDA |
| Sparse / BM25 | `Qdrant/bm25` (FastEmbed) | stemmer off; **WordNet lemmatization** applied instead |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | lightweight industry-standard cross-encoder |

> **Model hosting:** models are currently **downloaded at startup** when the service is
> hosted. The plan is to later serve them via **Hugging Face Inference Endpoints**
> directly instead of loading them in-process.

---

## Stack

- **FastAPI** — HTTP layer
- **Qdrant** — vector DB (named dense + sparse vectors, native RRF)
- **SentenceTransformers** — dense embeddings + cross-encoder
- **FastEmbed + NLTK** — sparse BM25 with lemmatization
- **PyTorch (CUDA 12.8)** — GPU runtime
- **yt-dlp + youtube-transcript-api** — metadata + transcripts
- **LangChain text splitters** — `RecursiveCharacterTextSplitter` (chunk 800 / overlap 200)

---

## Setup

```bash
uv sync                      # installs deps (incl. torch cu128)
cp .env.example .env         # set RETRIEVAL_API_KEY, QDRANT_URL, etc.
uv run python main.py        # serves on :9000
```

First start downloads the models and NLTK assets (slow); later starts are fast.

### Configuration (`.env`)

| Var | Default | Purpose |
|-----|---------|---------|
| `RETRIEVAL_API_KEY` | — | Shared secret for `X-API-Key`. Unset → auth disabled (dev only) |
| `ENVIRONMENT` | `dev` | `dev*` enables hot reload |
| `QDRANT_URL` | — | Qdrant server URL; unset → embedded on-disk store |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | dense model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranker |

See [`app/config.py`](app/config.py) for the full list (chunking, candidate/top-K, RRF).

---

## Testing

Run the service, then open [`notebooks/test_endpoints.ipynb`](notebooks/test_endpoints.ipynb)
— it exercises ingestion, both retrieval modes, and the API-key header.
