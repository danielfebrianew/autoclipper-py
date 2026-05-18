from . import config, heatmap, llm, metadata, transcript


async def generate(
    youtube_url: str,
    max_clips: int = config.DEFAULT_MAX_CLIPS,
    min_duration: int = config.DEFAULT_MIN_DURATION,
    max_duration: int = config.DEFAULT_MAX_DURATION,
    language: str = config.DEFAULT_LANGUAGE,
) -> dict:
    meta = metadata.fetch(youtube_url)

    transcript_text, _ = transcript.fetch(meta["video_id"], language)

    heatmap_data = heatmap.analyze(meta.get("heatmap_raw") or [])

    result = await llm.analyze(
        metadata=meta,
        transcript_text=transcript_text,
        heatmap_text=heatmap_data["formatted_text"],
        max_clips=max_clips,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    # Inject source metadata ke output
    result["video_title"] = result.get("video_title") or meta["title"]
    result["video_duration"] = result.get("video_duration") or meta["duration_formatted"]
    result["video_duration_seconds"] = meta["duration_seconds"]
    result["source_url"] = youtube_url
    result["heatmap_available"] = heatmap_data["available"]

    return result
