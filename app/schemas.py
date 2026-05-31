from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


# --- Ingest ---------------------------------------------------------------

class IngestRequest(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=1)
    user_id: str | None = None


class IngestItemResult(BaseModel):
    url: str
    success: bool
    video_id: str | None = None
    chunks_indexed: int = 0
    already_indexed: bool = False
    metadata: dict[str, Any] | None = None
    error: str | None = None


class IngestResponse(BaseModel):
    results: list[IngestItemResult]


# --- Retrieve -------------------------------------------------------------

class RetrieveRequest(BaseModel):
    mode: Literal["chunks", "metadata"] = "chunks"
    query: str | None = None
    video_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=200)
    user_id: str | None = None
    session_id: str | None = None

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


class VideoMetadata(BaseModel):
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
    metadata: list[VideoMetadata] = []
