from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
)


def prepare_documents(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents = []
    for chunk in transcript:
        start = chunk["start"]
        duration = chunk["duration"]
        documents.append(
            {
                "text": chunk["text"],
                "start_time": start,
                "duration": duration,
                "end_time": round(start + duration, 2),
            }
        )
    return documents


def build_transcript_windows(
    documents: list[dict[str, Any]],
    window_size: int = settings.WINDOW_SIZE,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None

    for doc in documents:
        if current_start is None:
            current_start = doc["start_time"]
        current_text += " " + doc["text"]
        current_end = doc["end_time"]
        if len(current_text) >= window_size:
            windows.append(
                {
                    "text": current_text.strip(),
                    "start_time": current_start,
                    "end_time": current_end,
                }
            )
            current_text = ""
            current_start = None

    if current_text:
        windows.append(
            {
                "text": current_text.strip(),
                "start_time": current_start,
                "end_time": current_end,
            }
        )
    return windows


def split_windows_into_chunks(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    langchain_docs = [
        Document(
            page_content=w["text"],
            metadata={"start_time": w["start_time"], "end_time": w["end_time"]},
        )
        for w in windows
    ]
    split_docs = _text_splitter.split_documents(langchain_docs)
    return [
        {
            "text": d.page_content,
            "start_time": d.metadata["start_time"],
            "end_time": d.metadata["end_time"],
        }
        for d in split_docs
    ]


def transcript_to_chunks(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents = prepare_documents(transcript)
    windows = build_transcript_windows(documents)
    return split_windows_into_chunks(windows)
