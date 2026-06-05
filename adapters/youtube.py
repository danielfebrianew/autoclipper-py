"""YouTube adapter — wraps clip_generator/metadata.py and transcript.py."""

from dataclasses import dataclass, field

from core.exceptions import TranscriptUnavailableError


@dataclass
class VideoMetadata:
    video_id: str
    title: str
    channel: str
    duration_seconds: int
    duration_formatted: str
    heatmap_raw: list
    chapters: list
    view_count: int | None = None
    upload_date: str = ""


@dataclass
class TranscriptResult:
    text: str
    snippets: list


class YouTubeAdapter:
    def fetch_metadata(self, youtube_url: str) -> VideoMetadata:
        from clip_generator import metadata
        raw = metadata.fetch(youtube_url)
        return VideoMetadata(
            video_id=raw["video_id"],
            title=raw["title"],
            channel=raw["channel"],
            duration_seconds=raw["duration_seconds"],
            duration_formatted=raw["duration_formatted"],
            heatmap_raw=raw.get("heatmap_raw") or [],
            chapters=raw.get("chapters") or [],
            view_count=raw.get("view_count"),
            upload_date=raw.get("upload_date", ""),
        )

    def fetch_transcript(self, video_id: str, language: str = "id") -> TranscriptResult:
        from clip_generator import transcript
        try:
            text, snippets = transcript.fetch(video_id, language)
        except ValueError as e:
            raise TranscriptUnavailableError(str(e)) from e
        return TranscriptResult(text=text, snippets=snippets)
