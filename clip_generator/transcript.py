from youtube_transcript_api import YouTubeTranscriptApi

from . import config

_api = YouTubeTranscriptApi()


def fetch(video_id: str, language: str = config.DEFAULT_LANGUAGE) -> tuple[str, list]:
    """Return (formatted_text, raw_snippets)."""
    try:
        fetched = _api.fetch(video_id, languages=[language, "en", "id"])
        snippets = list(fetched)
    except Exception:
        try:
            transcript_list = _api.list(video_id)
            transcript = transcript_list.find_transcript([language, "en", "id"])
            fetched = transcript.fetch()
            snippets = list(fetched)
        except Exception as e:
            raise ValueError(f"Transcript tidak tersedia untuk video ini: {e}")

    formatted = _format(snippets)
    return formatted, snippets


def _format(snippets: list) -> str:
    lines = []
    total_words = 0
    for seg in snippets:
        start = int(getattr(seg, "start", 0))
        text = getattr(seg, "text", "").strip().replace("\n", " ")
        m, s = divmod(start, 60)
        lines.append(f"[{m:02d}:{s:02d}] {text}")
        total_words += len(text.split())
        if total_words >= config.MAX_TRANSCRIPT_WORDS:
            lines.append("[... transcript dipotong karena terlalu panjang ...]")
            break
    return "\n".join(lines)
