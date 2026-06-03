"""
Timestamp validation and snapping.

After the LLM proposes clip boundaries, this module audits each clip against
the raw transcript snippets to ensure:
  1. No sentence is cut mid-speech at the start or end.
  2. start_seconds snaps backward to the nearest sentence boundary before
     the first word of the clip's transcript_excerpt.
  3. end_seconds snaps forward to the nearest sentence boundary after the
     last word of the clip's transcript_excerpt.
  4. A minimum silence gap (PAD_SEC) is enforced between the boundary and
     the nearest speech segment.

All adjustments are logged so the caller can see what changed.
"""

import re
import logging

log = logging.getLogger("clip_generator.timestamps")

# How many seconds of breathing room we want outside the last/first word.
PAD_SEC = 1.5
# Max seconds we'll shift a boundary to find a clean cut point.
MAX_SNAP_SEC = 8.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _snippet_words(snippet) -> list[str]:
    text = getattr(snippet, "text", "") or ""
    return _normalize(text).split()


def _excerpt_boundary_words(excerpt: str, n: int = 6) -> tuple[list[str], list[str]]:
    """Return first-n and last-n significant words of the transcript excerpt."""
    words = _normalize(excerpt).split()
    words = [w for w in words if len(w) > 2]  # skip filler like 'di', 'ke', 'ya'
    return words[:n], words[-n:]


def _find_first_match(snippets: list, target_words: list[str], after_sec: float) -> float | None:
    """
    Find the start_time of the earliest snippet (at or after after_sec) whose
    text contains any of the target_words.
    """
    for snip in snippets:
        start = float(getattr(snip, "start", 0))
        if start < after_sec - MAX_SNAP_SEC:
            continue
        text = _normalize(getattr(snip, "text", ""))
        if any(w in text for w in target_words):
            return start
    return None


def _find_last_match(snippets: list, target_words: list[str], before_sec: float) -> float | None:
    """
    Find the end of the latest snippet (at or before before_sec) whose text
    contains any of the target_words. Returns snippet.start + snippet.duration.
    """
    best = None
    for snip in snippets:
        start = float(getattr(snip, "start", 0))
        dur   = float(getattr(snip, "duration", 2.0))
        end   = start + dur
        if start > before_sec + MAX_SNAP_SEC:
            continue
        text = _normalize(getattr(snip, "text", ""))
        if any(w in text for w in target_words):
            best = end
    return best


def _sentence_end_after(snippets: list, from_sec: float, max_look: float) -> float | None:
    """
    After from_sec, find the first gap >= PAD_SEC between consecutive snippets
    (a natural silence = sentence boundary). Returns the gap midpoint.
    """
    relevant = [
        (float(getattr(s, "start", 0)), float(getattr(s, "duration", 2.0)))
        for s in snippets
        if float(getattr(s, "start", 0)) >= from_sec - 1.0
    ]
    relevant.sort()
    for i in range(len(relevant) - 1):
        gap_start = relevant[i][0] + relevant[i][1]
        gap_end   = relevant[i + 1][0]
        gap       = gap_end - gap_start
        if gap >= PAD_SEC and gap_start >= from_sec:
            return gap_start + min(gap * 0.3, 0.5)  # cut just inside the silence
        if gap_start > from_sec + max_look:
            break
    return None


def _sentence_start_before(snippets: list, from_sec: float, max_look: float) -> float | None:
    """
    Before from_sec, find the last gap >= PAD_SEC between consecutive snippets.
    Returns the snippet start immediately after the gap (= clean sentence start).
    """
    relevant = [
        (float(getattr(s, "start", 0)), float(getattr(s, "duration", 2.0)))
        for s in snippets
        if float(getattr(s, "start", 0)) <= from_sec + 1.0
    ]
    relevant.sort()
    best = None
    for i in range(len(relevant) - 1):
        gap_start = relevant[i][0] + relevant[i][1]
        gap_end   = relevant[i + 1][0]
        gap       = gap_end - gap_start
        if gap >= PAD_SEC and gap_end <= from_sec:
            best = gap_end  # start of next sentence after silence
        if gap_start < from_sec - max_look:
            continue
    return best


def snap_clip(clip: dict, snippets: list, video_duration: float) -> dict:
    """
    Validate and snap a single clip's timestamps against raw transcript snippets.
    Returns a new clip dict with adjusted start_seconds / end_seconds / duration_seconds,
    plus a 'timestamp_adjustments' list describing what changed.
    """
    clip = dict(clip)
    adjustments = []

    orig_start = float(clip.get("start_seconds") or 0)
    orig_end   = float(clip.get("end_seconds")   or 0)
    excerpt    = clip.get("transcript_excerpt") or ""
    end_cue    = clip.get("end_cue") or ""

    new_start = orig_start
    new_end   = orig_end

    if not snippets or not excerpt:
        clip["timestamp_adjustments"] = adjustments
        return clip

    first_words, last_words = _excerpt_boundary_words(excerpt)

    # end_cue overrides last_words when available — it's a verbatim quote of the
    # closing sentence, so it's more reliable than the tail of a potentially
    # paraphrased excerpt.
    end_anchor_words, end_anchor_label = (
        (_excerpt_boundary_words(end_cue)[1], f"end_cue '{end_cue[:40]}…'")
        if end_cue
        else (last_words, f"kata terakhir excerpt '{last_words[-1] if last_words else ''}'")
    )

    # --- Snap start ---
    if first_words:
        match_start = _find_first_match(snippets, first_words, orig_start - MAX_SNAP_SEC)
        if match_start is not None:
            clean = _sentence_start_before(snippets, match_start, MAX_SNAP_SEC)
            candidate = clean if clean is not None else max(0.0, match_start - PAD_SEC)
            if abs(candidate - orig_start) > 0.5:
                adjustments.append(
                    f"start {orig_start:.1f}s → {candidate:.1f}s "
                    f"(kata pertama '{first_words[0]}' ditemukan di {match_start:.1f}s)"
                )
                new_start = candidate
        else:
            log.warning(
                "clip %s: kata pertama excerpt tidak ditemukan di transcript dekat %.1fs",
                clip.get("clip_id"), orig_start,
            )

    # --- Snap end (prefer end_cue, fallback to last_words of excerpt) ---
    if end_anchor_words:
        # Search window biased forward: end might be later than LLM guessed
        match_end = _find_last_match(snippets, end_anchor_words, orig_end + MAX_SNAP_SEC)
        if match_end is not None:
            clean = _sentence_end_after(snippets, match_end, MAX_SNAP_SEC)
            candidate = clean if clean is not None else min(video_duration, match_end + PAD_SEC)
            # Always accept a forward extension — only require 0.5s threshold for reductions
            if candidate > orig_end or abs(candidate - orig_end) > 0.5:
                adjustments.append(
                    f"end {orig_end:.1f}s → {candidate:.1f}s ({end_anchor_label} berakhir di {match_end:.1f}s)"
                )
                new_end = candidate
        else:
            log.warning(
                "clip %s: %s tidak ditemukan di transcript dekat %.1fs — end tidak disesuaikan",
                clip.get("clip_id"), end_anchor_label, orig_end,
            )

    # --- Sanity clamps ---
    new_start = max(0.0, new_start)
    new_end   = min(video_duration, new_end)
    if new_end - new_start < 5.0:
        log.warning(
            "clip %s: durasi setelah snap terlalu pendek (%.1fs), kembali ke original",
            clip.get("clip_id"), new_end - new_start,
        )
        new_start = orig_start
        new_end   = orig_end
        adjustments.append("snap dibatalkan — hasil terlalu pendek, pakai timestamp LLM")

    def _fmt(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}"

    clip["start_seconds"]  = int(new_start)
    clip["end_seconds"]    = int(new_end)
    clip["start_time"]     = _fmt(new_start)
    clip["end_time"]       = _fmt(new_end)
    clip["duration_seconds"] = int(new_end - new_start)
    clip["timestamp_adjustments"] = adjustments

    return clip


def validate_and_snap(result: dict, snippets: list) -> dict:
    """
    Run snap_clip on every clip in the LLM result dict.
    Mutates result in-place and returns it.
    """
    video_duration = float(result.get("video_duration_seconds") or 0)
    clips = result.get("clips") or []
    total_adjusted = 0

    for i, clip in enumerate(clips):
        snapped = snap_clip(clip, snippets, video_duration)
        adj = snapped.get("timestamp_adjustments") or []
        if adj:
            total_adjusted += 1
            log.info(
                "clip %s — %d penyesuaian timestamp: %s",
                clip.get("clip_id"), len(adj), " | ".join(adj),
            )
        clips[i] = snapped

    log.info("%d dari %d clip disesuaikan timestampnya", total_adjusted, len(clips))
    result["clips"] = clips
    return result
