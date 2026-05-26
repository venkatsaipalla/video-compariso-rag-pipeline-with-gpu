from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class IngestRequest(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=1)


class VideoMetadata(BaseModel):
    video_id: str
    title: str | None = None
    channel: str | None = None
    duration: int | None = None
    upload_date: str | None = None
    view_count: int | None = None
    like_count: int | None = None


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
