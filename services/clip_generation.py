"""
ClipGenerationService — orchestrates YouTube metadata, transcript, LLM analysis,
and timestamp snapping into a final clips JSON result.

Replaces clip_generator/__init__.py's generate() with a class that has
injected dependencies (youtube adapter, config).
"""

from adapters.youtube import YouTubeAdapter, VideoMetadata, TranscriptResult
from core.exceptions import TranscriptUnavailableError, LLMError
from processing.logger import get_logger

log = get_logger("services.clip_generation")


class ClipGenerationService:
    def __init__(self, youtube: YouTubeAdapter | None = None):
        self.youtube = youtube or YouTubeAdapter()

    async def generate(
        self,
        youtube_url: str,
        max_clips: int = 10,
        min_duration: int = 20,
        max_duration: int = 90,
        language: str = "id",
    ) -> dict:
        log.info("Fetching metadata: %s", youtube_url)
        meta = self.youtube.fetch_metadata(youtube_url)

        log.info("Fetching transcript (lang=%s)...", language)
        try:
            transcript = self.youtube.fetch_transcript(meta.video_id, language)
        except TranscriptUnavailableError:
            raise ValueError(f"Transcript tidak tersedia untuk video ini: {meta.video_id}")

        result = await self._analyze(meta, transcript, max_clips, min_duration, max_duration)

        result["video_title"]            = result.get("video_title") or meta.title
        result["video_duration"]         = result.get("video_duration") or meta.duration_formatted
        result["video_duration_seconds"] = meta.duration_seconds
        result["source_url"]             = youtube_url
        result["heatmap_available"]      = bool(meta.heatmap_raw)

        self._snap_timestamps(result, transcript.snippets)
        return result

    # ── private ───────────────────────────────────────────────────────────────

    async def _analyze(
        self,
        meta: VideoMetadata,
        transcript: TranscriptResult,
        max_clips: int,
        min_duration: int,
        max_duration: int,
    ) -> dict:
        from clip_generator import heatmap as heatmap_mod, llm

        heatmap_data = heatmap_mod.analyze(meta.heatmap_raw)

        meta_dict = {
            "video_id":           meta.video_id,
            "title":              meta.title,
            "channel":            meta.channel,
            "duration_seconds":   meta.duration_seconds,
            "duration_formatted": meta.duration_formatted,
            "heatmap_raw":        meta.heatmap_raw,
            "chapters":           meta.chapters,
        }

        try:
            result = await llm.analyze(
                metadata=meta_dict,
                transcript_text=transcript.text,
                heatmap_text=heatmap_data["formatted_text"],
                max_clips=max_clips,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        except Exception as e:
            raise LLMError(str(e)) from e

        result["heatmap_available"] = heatmap_data["available"]
        return result

    def _snap_timestamps(self, result: dict, snippets: list) -> None:
        from clip_generator import timestamps
        timestamps.validate_and_snap(result, snippets)
