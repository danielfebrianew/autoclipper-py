import json
import subprocess


def fetch(youtube_url: str) -> dict:
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-playlist", youtube_url],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise ValueError(f"yt-dlp gagal: {result.stderr.strip()[:300]}")

    data = json.loads(result.stdout)

    duration = int(data.get("duration") or 0)
    heatmap_raw = data.get("heatmap") or []

    chapters = []
    for ch in data.get("chapters") or []:
        chapters.append({
            "title": ch.get("title", ""),
            "start_seconds": int(ch.get("start_time", 0)),
            "end_seconds": int(ch.get("end_time", 0)),
        })

    return {
        "video_id": data.get("id", ""),
        "title": data.get("title", ""),
        "channel": data.get("channel") or data.get("uploader", ""),
        "duration_seconds": duration,
        "duration_formatted": _fmt(duration),
        "view_count": data.get("view_count"),
        "upload_date": data.get("upload_date", ""),
        "heatmap_raw": heatmap_raw,
        "chapters": chapters,
    }


def _fmt(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
