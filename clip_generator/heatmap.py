from . import config


def analyze(heatmap_raw: list[dict]) -> dict:
    if not heatmap_raw:
        return {"available": False, "peaks": [], "formatted_text": "Tidak tersedia."}

    values = [float(h.get("value", 0)) for h in heatmap_raw]
    max_val = max(values) if values else 1.0
    min_val = min(values) if values else 0.0
    spread = max_val - min_val or 1.0

    normalized = [(v - min_val) / spread for v in values]

    peaks = []
    for i in range(1, len(normalized) - 1):
        if normalized[i] >= config.HEATMAP_PEAK_THRESHOLD and normalized[i] >= normalized[i - 1] and normalized[i] >= normalized[i + 1]:
            seg = heatmap_raw[i]
            intensity = normalized[i]
            tag = "HIGH" if intensity >= config.HEATMAP_HIGH_THRESHOLD else "MEDIUM"
            peaks.append({
                "start_seconds": int(seg.get("start_time", 0)),
                "end_seconds": int(seg.get("end_time", 0)),
                "start_formatted": _fmt(int(seg.get("start_time", 0))),
                "end_formatted": _fmt(int(seg.get("end_time", 0))),
                "intensity": round(intensity, 2),
                "tag": tag,
            })

    peaks.sort(key=lambda x: x["intensity"], reverse=True)
    for i, p in enumerate(peaks):
        p["rank"] = i + 1

    formatted = _format_text(peaks)
    return {"available": True, "peaks": peaks, "formatted_text": formatted}


def _format_text(peaks: list[dict]) -> str:
    if not peaks:
        return "Heatmap tersedia tapi tidak ada peak signifikan."
    lines = ["DATA HEATMAP (Most Replayed oleh penonton YouTube):"]
    for p in peaks[:15]:
        icon = "🔴" if p["tag"] == "HIGH" else "🟡"
        lines.append(
            f"{icon} PEAK #{p['rank']} [{p['start_formatted']}-{p['end_formatted']}]: "
            f"intensity {p['intensity']} ({p['tag']})"
        )
    return "\n".join(lines)


def _fmt(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"
