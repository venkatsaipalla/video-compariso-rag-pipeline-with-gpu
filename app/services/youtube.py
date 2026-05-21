from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

_ytt_api = YouTubeTranscriptApi()


def extract_video_metadata(url: str) -> dict[str, Any]:
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "description": info.get("description"),
        "channel": info.get("channel"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "tags": info.get("tags"),
    }


_PREFERRED_LANGS = ("en", "en-US", "en-GB", "en-IN", "en-CA", "en-AU")


def extract_transcript(video_id: str) -> list[dict[str, Any]]:
    # Try preferred English variants directly first.
    try:
        return _ytt_api.fetch(video_id, languages=list(_PREFERRED_LANGS)).to_raw_data()
    except Exception:
        pass

    # Fallback: enumerate, prefer any English variant, else translate to English.
    transcript_list = _ytt_api.list(video_id)
    for t in transcript_list:
        if t.language_code.lower().startswith("en"):
            return t.fetch().to_raw_data()
    for t in transcript_list:
        if getattr(t, "is_translatable", False):
            return t.translate("en").fetch().to_raw_data()
    for t in transcript_list:
        return t.fetch().to_raw_data()
    raise RuntimeError(f"no transcripts available for video_id={video_id}")
