# Architecture — High-Level Session & Interaction Flow

```
                       ┌──────────────────────────┐
                       │       Frontend (UI)      │
                       └─────────────┬────────────┘
                                     │
                       ┌─────────────┴───────────────────────────────┐
                       │                                             │
                  (1) Google sign-in                          (later) chat msgs
                       │                                             │
                       ▼                                             ▼
        ╔══════════════════════════════════════════════════════════════════════╗
        ║                    BACKEND  (cognitive layer · ADK)                  ║
        ║                                                                      ║
        ║   /auth ──► verify Google token, issue app JWT                       ║
        ║                                                                      ║
        ║   /init  ──► (2) POST /ingest ─────────────────────────► GPU repo    ║
        ║              (waits until ingestion completes)                       ║
        ║              (3) create ADK session ──► session_id                   ║
        ║              (4) return session_id ────────────────────► Frontend    ║
        ║                                                                      ║
        ║   /chat  ──► receives { session_id, user_msg }                       ║
        ║              ┌────────────────────────────────────────┐              ║
        ║              │   Multi-Agent Ecosystem (Google ADK)   │              ║
        ║              │                                        │              ║
        ║              │      root ─► router ─► [ rag |         │              ║
        ║              │                          analysis ]    │              ║
        ║              │                  ─► reducer ─► final   │              ║
        ║              │                                        │              ║
        ║              │   - rag agents: planner → retriever →  │              ║
        ║              │     grader → packer                    │              ║
        ║              │   - analysis agents: comparator,       │              ║
        ║              │     timeline, virality, summarizer …   │              ║
        ║              │   - final agent: synthesizes answer    │              ║
        ║              │     with citations                     │              ║
        ║              └────────────────────────────────────────┘              ║
        ║                          │                                           ║
        ║                          │ (5) if retrieval needed                   ║
        ║                          ▼                                           ║
        ║              POST /retrieve ──────────────────────────► GPU repo    ║
        ║                                                                      ║
        ║   ◄── (6) final answer + citations ───── Frontend                    ║
        ╚══════════════════════════════════════════════════════════════════════╝

                                                       ▲
                                  HTTPS · X-API-Key    │
                                  + user_id            │
                                  (+ session_id on     │
                                   /retrieve)          │
                                                       │
        ╔══════════════════════════════════════════════╧═══════════════════════╗
        ║                          GPU  REPO (this one)                        ║
        ║                                                                      ║
        ║   /ingest   ── url ─► yt-dlp metadata                                ║
        ║                 │     youtube-transcript-api                         ║
        ║                 │     recursive chunking (window + overlap)          ║
        ║                 │     dense embed (BGE) + sparse encode (BM25)       ║
        ║                 └──► Qdrant upsert  (named dense + sparse vectors)   ║
        ║                                                                      ║
        ║   /retrieve ── query ─► dense search ┐                               ║
        ║                         sparse BM25  ├─► Qdrant RRF fusion           ║
        ║                                      ┘        │                      ║
        ║                                               ▼                      ║
        ║                                  cross-encoder rerank ─► top-K       ║
        ║                                                                      ║
        ║                          ┌────────────┐                              ║
        ║                          │   Qdrant   │  (dense + sparse, per chunk) ║
        ║                          └────────────┘                              ║
        ╚══════════════════════════════════════════════════════════════════════╝


─────────────────────────────  Session lifecycle  ─────────────────────────────

  (1) Frontend     →  Google sign-in (OAuth)         →  Backend /auth
  (2) Frontend     →  POST /init { urls }            →  Backend
                      Backend  → POST /ingest        →  GPU repo
                      GPU      → ingest + index in Qdrant
  (3) Backend      →  ADK creates session            →  session_id
  (4) Backend      →  returns { session_id }         →  Frontend
  (5) Frontend     →  POST /chat { session_id, msg } →  Backend
                      Backend agents run; if retrieval needed:
                      Backend  → POST /retrieve { session_id, user_id, query }
                              →  GPU repo (hybrid + RRF + rerank)
  (6) Backend      →  final synthesized answer       →  Frontend
```

## Notes

- The **Frontend** never talks to the GPU repo directly. All GPU calls are
  proxied by the Backend, which is the only holder of the `X-API-Key`.
- **Ingestion is a one-shot** owned by `/init` on the Backend. No agent is
  allowed to trigger ingestion mid-session — video set is fixed when the
  session is created.
- **Session state** (history, video_ids, scratch) lives in the Backend's ADK
  session store; the GPU repo is stateless w.r.t. sessions and only uses
  `user_id` / `session_id` for log-context binding.
- **Logging context** propagates: Backend binds `user_id` + `session_id` per
  request, and forwards both to the GPU repo, which re-binds them so every
  log line on both sides is correlatable.
