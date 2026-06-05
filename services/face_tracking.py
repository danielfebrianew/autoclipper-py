"""
Face tracking services — stateful wrappers around reframe.py logic.

FaceTracker:     single-speaker, used by VideoProcessingService single mode.
DualFaceTracker: split-screen, used by VideoProcessingService split mode.

Both classes are testable with mock Face lists without needing a video file.
"""

import numpy as np

from core.models import Face, FocusTrackerState
from processing import config
from processing.face import pick_best_face, match_face_by_center


class FaceTracker:
    """
    Stateful single-speaker face tracker.
    One instance per clip — call update() per sampled frame.
    """

    def __init__(self, src_w: int, src_h: int, src_fps: float):
        self.src_w = src_w
        self.src_h = src_h
        self.src_fps = max(src_fps, 1.0)

        crop_w = int(src_h * 9 / 16)
        self._clamp_min = crop_w / 2
        self._clamp_max = src_w - crop_w / 2
        self._default_cx = src_w / 2
        self._match_distance_px = max(
            crop_w * config.FOCUS_MATCH_DISTANCE_RATIO, config.CROP_MIN_DEADZONE_PX
        )
        self._min_lock_frames = max(1, int(self.src_fps * config.FOCUS_MIN_LOCK_SEC))
        self._confirm_frames = int(self.src_fps * config.FOCUS_SWITCH_CONFIRM_SEC)
        self._lost_grace_frames = int(self.src_fps * config.FOCUS_LOST_GRACE_SEC)

        self._state = FocusTrackerState()
        self._keyframes: list[tuple] = []
        self._hard_cut_frame_set: set[int] = set()
        self._smooth_focus_changes = 0
        self._source_cut_resets = 0

    def _clamp(self, cx: float) -> float:
        return float(np.clip(cx, self._clamp_min, self._clamp_max))

    def _add_keyframe(self, frame_idx: int, cx: float, hard: bool = False) -> None:
        self._keyframes.append((frame_idx, self._clamp(cx), hard))
        if hard:
            self._hard_cut_frame_set.add(frame_idx)

    def update(self, frame_idx: int, faces: list[Face],
               is_scene_cut: bool = False, asd_scores: dict | None = None) -> None:
        state = self._state
        frame_asd = asd_scores.get(frame_idx) if asd_scores else None
        best_face = pick_best_face(faces, frame_asd)
        default_cx = self._default_cx
        match_dist = self._match_distance_px

        if is_scene_cut and frame_idx > state.prev_sample_frame:
            self._source_cut_resets += 1
            state.reset_pending()
            state.lock_until_frame = -1
            state.current_cx = best_face.cx if best_face else default_cx
            state.current_area = best_face.area if best_face else 0.0
            state.last_seen_frame = frame_idx if best_face else -(10**9)
            state.lock(frame_idx, self._min_lock_frames)
            self._add_keyframe(frame_idx, state.current_cx)
            state.prev_sample_frame = frame_idx
            return

        if state.current_cx is None:
            state.current_cx = best_face.cx if best_face else default_cx
            state.current_area = best_face.area if best_face else 0.0
            if best_face:
                state.last_seen_frame = frame_idx
            state.lock(frame_idx, self._min_lock_frames)
            self._add_keyframe(frame_idx, state.current_cx)
            state.prev_sample_frame = frame_idx
            return

        current_face = match_face_by_center(faces, state.current_cx, match_dist)
        current_visible = current_face is not None
        current_lost = state.lost_too_long(frame_idx, self._lost_grace_frames)

        if current_visible:
            state.current_cx = current_face.cx
            state.current_area = current_face.area
            state.last_seen_frame = frame_idx
            current_lost = False
        elif not faces and current_lost:
            state.current_cx = default_cx
            state.current_area = 0.0
            state.reset_pending()

        other_faces = [f for f in faces if abs(f.cx - state.current_cx) > match_dist]
        candidate = pick_best_face(other_faces, frame_asd) if other_faces else (
            best_face if best_face and abs(best_face.cx - state.current_cx) > match_dist else None
        )

        if candidate is not None and not state.is_locked(frame_idx):
            size_wins = (state.current_area <= 0
                         or candidate.area >= state.current_area * config.FOCUS_SWITCH_AREA_RATIO)
            if current_lost or size_wins:
                if state.pending_cx is None or abs(candidate.cx - state.pending_cx) > match_dist:
                    state.pending_cx = candidate.cx
                    state.pending_since_frame = frame_idx
                else:
                    state.pending_cx = candidate.cx
                if state.pending_since_frame is None:
                    state.pending_since_frame = frame_idx
                if frame_idx - state.pending_since_frame >= self._confirm_frames:
                    if abs(candidate.cx - state.current_cx) > match_dist * 0.5:
                        self._smooth_focus_changes += 1
                    state.current_cx = candidate.cx
                    state.current_area = candidate.area
                    state.last_seen_frame = frame_idx
                    state.lock(frame_idx, self._min_lock_frames)
            elif current_visible:
                state.reset_pending()
        elif current_visible:
            state.reset_pending()

        self._add_keyframe(frame_idx, state.current_cx)
        state.prev_sample_frame = frame_idx

    def get_keyframes(self) -> list[tuple]:
        return list(self._keyframes)

    def get_hard_cut_frames(self) -> set[int]:
        return set(self._hard_cut_frame_set)

    def stats(self) -> dict:
        return {
            "source_cut_resets": self._source_cut_resets,
            "smooth_focus_changes": self._smooth_focus_changes,
        }


class DualFaceTracker:
    """
    Split-screen variant: track left and right speaker independently.
    Falls back to single full-width crop when only one face is detected.
    """

    def __init__(self, src_w: int, src_h: int, src_fps: float):
        self.src_w = src_w
        self.src_h = src_h
        self.src_fps = max(src_fps, 1.0)

        panel_w = int(src_h * 9 / 16) // 2
        crop_w_single = panel_w * 2

        self._panel_w = panel_w
        self._clamp_left_min  = panel_w / 2
        self._clamp_left_max  = src_w / 2
        self._clamp_right_min = src_w / 2
        self._clamp_right_max = src_w - panel_w / 2
        self._clamp_single_min = crop_w_single / 2
        self._clamp_single_max = src_w - crop_w_single / 2

        self._cx_left   = src_w * 0.25
        self._cx_right  = src_w * 0.75
        self._cx_single = src_w / 2

        self._raw_left:     list[tuple[int, float]] = []
        self._raw_right:    list[tuple[int, float]] = []
        self._raw_single:   list[tuple[int, float]] = []
        self._raw_is_split: list[tuple[int, bool]]  = []

    def update(self, frame_idx: int, faces: list[Face],
               asd_scores: dict | None = None) -> None:
        frame_asd = asd_scores.get(frame_idx) if asd_scores else None
        mid = self.src_w / 2

        left_faces  = [f for f in faces if f.cx <  mid]
        right_faces = [f for f in faces if f.cx >= mid]

        best_left  = pick_best_face(left_faces,  frame_asd)
        best_right = pick_best_face(right_faces, frame_asd)
        two_faces  = best_left is not None and best_right is not None

        if not two_faces:
            single = best_left or best_right
            if single is not None:
                self._cx_single = float(np.clip(
                    single.cx, self._clamp_single_min, self._clamp_single_max
                ))
            self._raw_left.append((frame_idx, self._cx_left))
            self._raw_right.append((frame_idx, self._cx_right))
            self._raw_single.append((frame_idx, self._cx_single))
            self._raw_is_split.append((frame_idx, False))
            return

        self._cx_left  = float(np.clip(best_left.cx,  self._clamp_left_min,  self._clamp_left_max))
        self._cx_right = float(np.clip(best_right.cx, self._clamp_right_min, self._clamp_right_max))

        self._raw_left.append((frame_idx, self._cx_left))
        self._raw_right.append((frame_idx, self._cx_right))
        self._raw_single.append((frame_idx, self._cx_single))
        self._raw_is_split.append((frame_idx, True))

    def get_raw(self) -> tuple[list, list, list, list]:
        """Return (raw_left, raw_right, raw_single, raw_is_split)."""
        return self._raw_left, self._raw_right, self._raw_single, self._raw_is_split

    def default_values(self) -> tuple[float, float, float]:
        return self.src_w * 0.25, self.src_w * 0.75, self.src_w / 2

    def clamp_bounds(self) -> dict:
        return {
            "left_min":   self._clamp_left_min,
            "left_max":   self._clamp_left_max,
            "right_min":  self._clamp_right_min,
            "right_max":  self._clamp_right_max,
            "single_min": self._clamp_single_min,
            "single_max": self._clamp_single_max,
        }
