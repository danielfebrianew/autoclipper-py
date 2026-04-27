from . import config

_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 608
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,64,&H00FFFFFF,&H0000FFFF,&H00000000,&HCC000000,-1,0,0,0,100,100,0,0,1,4,3,2,10,10,450,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

_WHITE  = "&H00FFFFFF"
_YELLOW = "&H0000FFFF"


def format_timestamp_ass(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def write_ass(segments_list, ass_path: str) -> None:
    def write_chunk(f, chunk):
        for i, active in enumerate(chunk):
            seg_start = active.start
            seg_end   = active.end
            if seg_end <= seg_start:
                seg_end = seg_start + 0.01
            parts = []
            for j, w in enumerate(chunk):
                word_text = w.word.strip().upper()
                if j == i:
                    parts.append(f"{{\\c{_YELLOW}}}{word_text}{{\\c{_WHITE}}}")
                else:
                    parts.append(word_text)
            text = " ".join(parts)
            f.write(
                f"Dialogue: 0,{format_timestamp_ass(seg_start)},"
                f"{format_timestamp_ass(seg_end)},Default,,0,0,0,,{text}\n"
            )

    all_words = [w for seg in segments_list for w in seg.words]
    n = config.MAX_WORDS_PER_SCREEN
    chunks = [all_words[i:i + n] for i in range(0, len(all_words), n)]

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(_ASS_HEADER)
        for chunk in chunks:
            write_chunk(f, chunk)
