import cv2
import numpy as np
from . import config
from .face import pick_best_face, match_face_by_center, sample_face_frame
from core.models import Face, FocusTrackerState


def _interpolate_targets_by_scene(keyframes, scene_cut_frames, total_frames, default_cx):
    if total_frames <= 0:
        return np.array([], dtype=float), set()

    raw_targets = np.full(total_frames, default_cx, dtype=float)
    hard_cut_frames = set()

    deduped = {}
    for entry in keyframes:
        frame_num, cx = entry[0], entry[1]
        hard = entry[2] if len(entry) > 2 else False
        if 0 <= frame_num < total_frames:
            deduped[int(frame_num)] = (float(cx), bool(hard))
            if hard:
                hard_cut_frames.add(int(frame_num))

    if not deduped:
        return raw_targets, hard_cut_frames

    sorted_keys = sorted(deduped.items())
    all_cuts = sorted(set(fn for fn in scene_cut_frames if 0 < fn < total_frames) | hard_cut_frames)
    segment_starts = [0] + all_cuts
    segment_ends   = all_cuts + [total_frames]

    def ease(t):
        return t * t * (3 - 2 * t)

    for seg_start, seg_end in zip(segment_starts, segment_ends):
        seg_keys = [(fn, cx) for fn, (cx, _) in sorted_keys if seg_start <= fn < seg_end]
        if not seg_keys:
            fill_value = raw_targets[seg_start - 1] if seg_start > 0 else default_cx
            raw_targets[seg_start:seg_end] = fill_value
            continue

        first_fn, first_cx = seg_keys[0]
        raw_targets[seg_start:min(first_fn + 1, seg_end)] = first_cx

        for idx in range(len(seg_keys) - 1):
            fn_a, cx_a = seg_keys[idx]
            fn_b, cx_b = seg_keys[idx + 1]
            span = fn_b - fn_a
            if span <= 0:
                continue
            for frame_num in range(fn_a, min(fn_b + 1, seg_end)):
                t = (frame_num - fn_a) / span
                raw_targets[frame_num] = cx_a + (cx_b - cx_a) * ease(t)

        last_fn, last_cx = seg_keys[-1]
        raw_targets[last_fn:seg_end] = last_cx

    return raw_targets, hard_cut_frames


def _apply_crop_smoothing(raw_targets, scene_cut_frames, crop_w, total_frames,
                          src_fps, hard_cut_frames=None):
    if total_frames <= 0:
        return raw_targets, {"hard_crop_jumps": 0}

    centers = np.empty(total_frames, dtype=float)
    centers[0] = raw_targets[0]

    cut_frames = set(fn for fn in scene_cut_frames if 0 < fn < total_frames)
    if hard_cut_frames:
        cut_frames |= hard_cut_frames

    deadzone_px = max(crop_w * config.CROP_DEADZONE_RATIO, config.CROP_MIN_DEADZONE_PX)
    max_step_px = max(1.0, config.CROP_MAX_SPEED_PX_PER_SEC / max(src_fps, 1.0))
    alpha = 1.0 - np.exp(-1.0 / max(src_fps * config.CROP_SMOOTHING_TAU_SEC, 1.0))
    hard_crop_jumps = 0

    for frame_num in range(1, total_frames):
        target_cx = raw_targets[frame_num]
        if frame_num in cut_frames:
            if abs(target_cx - centers[frame_num - 1]) > deadzone_px:
                hard_crop_jumps += 1
            centers[frame_num] = target_cx
            continue

        delta = target_cx - centers[frame_num - 1]
        if abs(delta) <= deadzone_px:
            centers[frame_num] = centers[frame_num - 1]
            continue

        desired_cx = target_cx - (np.sign(delta) * deadzone_px)
        step = float(np.clip((desired_cx - centers[frame_num - 1]) * alpha, -max_step_px, max_step_px))
        centers[frame_num] = centers[frame_num - 1] + step

    return centers, {"hard_crop_jumps": hard_crop_jumps}


# ── Scene cut detection helpers ────────────────────────────────────────────

class _SceneCutState:
    __slots__ = ("prev_gray", "prev_hist", "last_cut_frame")

    def __init__(self, min_gap_frames: int):
        self.prev_gray = None
        self.prev_hist = None
        self.last_cut_frame = -min_gap_frames


def _detect_scene_cut(frame, state: _SceneCutState, frame_idx: int, min_gap_frames: int) -> bool:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)

    is_cut = False
    if state.prev_gray is not None and state.prev_hist is not None:
        pixel_diff = float(np.mean(cv2.absdiff(gray, state.prev_gray)) / 255.0)
        hist_diff  = float(cv2.compareHist(state.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        score = 0.60 * hist_diff + 0.40 * pixel_diff

        normal_cut   = score >= config.SCENE_CUT_SCORE_THRESHOLD and hist_diff >= config.SCENE_CUT_HIST_THRESHOLD and pixel_diff >= config.SCENE_CUT_PIXEL_THRESHOLD
        strong_hist  = hist_diff >= max(0.45, config.SCENE_CUT_HIST_THRESHOLD * 2.5) and score >= config.SCENE_CUT_SCORE_THRESHOLD
        strong_pixel = pixel_diff >= max(0.22, config.SCENE_CUT_PIXEL_THRESHOLD * 2.0) and hist_diff >= config.SCENE_CUT_HIST_THRESHOLD * 0.75

        if (normal_cut or strong_hist or strong_pixel) and frame_idx - state.last_cut_frame >= min_gap_frames:
            state.last_cut_frame = frame_idx
            is_cut = True

    state.prev_gray = gray
    state.prev_hist = hist
    return is_cut


# ── Focus tracking helpers ─────────────────────────────────────────────────

def _update_focus(
    state: FocusTrackerState,
    faces: list[Face],
    frame_idx: int,
    is_scene_cut: bool,
    default_cx: float,
    clamp_min: float,
    clamp_max: float,
    min_lock_frames: int,
    confirm_frames: int,
    lost_grace_frames: int,
    match_distance_px: float,
    asd_scores: dict | None,
    keyframes: list,
    hard_cut_frame_set: set,
    smooth_focus_changes_ref: list,  # mutable single-element list as counter ref
) -> None:
    frame_asd = asd_scores.get(frame_idx) if asd_scores else None
    best_face = pick_best_face(faces, frame_asd)

    def clamp_cx(cx):
        return float(np.clip(cx, clamp_min, clamp_max))

    def add_keyframe(fn, cx, hard=False):
        fn = int(fn)
        keyframes.append((fn, clamp_cx(cx), hard))
        if hard:
            hard_cut_frame_set.add(fn)

    # Scene cut at a sampling frame — reset tracking
    if is_scene_cut and frame_idx > state.prev_sample_frame:
        state.reset_pending()
        state.lock_until_frame = -1
        state.current_cx = best_face.cx if best_face is not None else default_cx
        state.current_area = best_face.area if best_face is not None else 0.0
        state.last_seen_frame = frame_idx if best_face is not None else -(10**9)
        state.lock(frame_idx, min_lock_frames)
        add_keyframe(frame_idx, state.current_cx)
        state.prev_sample_frame = frame_idx
        return

    # First frame ever seen
    if state.current_cx is None:
        state.current_cx = best_face.cx if best_face is not None else default_cx
        state.current_area = best_face.area if best_face is not None else 0.0
        if best_face is not None:
            state.last_seen_frame = frame_idx
        state.lock(frame_idx, min_lock_frames)
        add_keyframe(frame_idx, state.current_cx)
        state.prev_sample_frame = frame_idx
        return

    # Track current subject
    current_face = match_face_by_center(faces, state.current_cx, match_distance_px)
    current_visible = current_face is not None
    current_lost_too_long = state.lost_too_long(frame_idx, lost_grace_frames)

    if current_visible:
        state.current_cx = current_face.cx
        state.current_area = current_face.area
        state.last_seen_frame = frame_idx
        current_lost_too_long = False
    elif not faces and current_lost_too_long:
        state.current_cx = default_cx
        state.current_area = 0.0
        state.reset_pending()

    # Candidate switch logic
    other_faces = [f for f in faces if abs(f.cx - state.current_cx) > match_distance_px]
    candidate = pick_best_face(other_faces, frame_asd) if other_faces else (
        best_face if best_face is not None and abs(best_face.cx - state.current_cx) > match_distance_px else None
    )

    can_switch = not state.is_locked(frame_idx)
    if candidate is not None and can_switch:
        size_wins = state.current_area <= 0 or candidate.area >= state.current_area * config.FOCUS_SWITCH_AREA_RATIO
        candidate_is_better = current_lost_too_long or size_wins

        if candidate_is_better:
            if state.pending_cx is None or abs(candidate.cx - state.pending_cx) > match_distance_px:
                state.pending_cx = candidate.cx
                state.pending_since_frame = frame_idx
            else:
                state.pending_cx = candidate.cx

            if state.pending_since_frame is None:
                state.pending_since_frame = frame_idx

            pending_age = frame_idx - state.pending_since_frame
            if pending_age >= confirm_frames:
                if abs(candidate.cx - state.current_cx) > match_distance_px * 0.5:
                    smooth_focus_changes_ref[0] += 1
                state.current_cx = candidate.cx
                state.current_area = candidate.area
                state.last_seen_frame = frame_idx
                state.lock(frame_idx, min_lock_frames)
        elif current_visible:
            state.reset_pending()
    elif current_visible:
        state.reset_pending()

    add_keyframe(frame_idx, state.current_cx)
    state.prev_sample_frame = frame_idx


# ── Public entry points ────────────────────────────────────────────────────

def compute_crop_centers(face_data, scene_cut_frames, src_w, src_h, total_frames, src_fps,
                         asd_scores: dict | None = None):
    """Legacy entry point used by tests or external callers."""
    crop_w = int(src_h * 9 / 16)
    half_crop = crop_w / 2
    default_cx = src_w / 2
    clamp_min = half_crop
    clamp_max = src_w - half_crop

    min_lock_frames   = int(src_fps * config.FOCUS_MIN_LOCK_SEC)
    confirm_frames    = int(src_fps * config.FOCUS_SWITCH_CONFIRM_SEC)
    lost_grace_frames = int(src_fps * config.FOCUS_LOST_GRACE_SEC)
    match_distance_px = max(crop_w * config.FOCUS_MATCH_DISTANCE_RATIO, config.CROP_MIN_DEADZONE_PX)

    scene_cuts = sorted(fn for fn in scene_cut_frames if 0 < fn < total_frames)
    cut_idx = 0

    state = FocusTrackerState()
    keyframes = []
    hard_cut_frame_set = set()
    smooth_focus_changes_ref = [0]
    source_cut_resets = 0

    def clamp_cx(cx):
        return float(np.clip(cx, clamp_min, clamp_max))

    def add_keyframe(fn, cx, hard=False):
        if total_frames <= 0:
            return
        fn = max(0, min(int(fn), total_frames - 1))
        keyframes.append((fn, clamp_cx(cx), hard))
        if hard:
            hard_cut_frame_set.add(fn)

    for sample in face_data:
        frame_num = int(sample["frame"])
        faces: list[Face] = sample["faces"]
        frame_asd = asd_scores.get(frame_num) if asd_scores else None
        best_face = pick_best_face(faces, frame_asd)

        source_cut_reset = False
        while cut_idx < len(scene_cuts) and scene_cuts[cut_idx] <= frame_num:
            if scene_cuts[cut_idx] > state.prev_sample_frame:
                source_cut_reset = True
            cut_idx += 1

        if source_cut_reset:
            source_cut_resets += 1
            state.reset_pending()
            state.current_cx = best_face.cx if best_face is not None else default_cx
            state.current_area = best_face.area if best_face is not None else 0.0
            state.last_seen_frame = frame_num if best_face is not None else -(10**9)
            state.lock(frame_num, min_lock_frames)
            add_keyframe(frame_num, state.current_cx)
            state.prev_sample_frame = frame_num
            continue

        if state.current_cx is None:
            state.current_cx = best_face.cx if best_face is not None else default_cx
            state.current_area = best_face.area if best_face is not None else 0.0
            if best_face is not None:
                state.last_seen_frame = frame_num
            state.lock(frame_num, min_lock_frames)
            add_keyframe(frame_num, state.current_cx)
            state.prev_sample_frame = frame_num
            continue

        current_face = match_face_by_center(faces, state.current_cx, match_distance_px)
        current_visible = current_face is not None
        current_lost_too_long = state.lost_too_long(frame_num, lost_grace_frames)

        if current_visible:
            state.current_cx = current_face.cx
            state.current_area = current_face.area
            state.last_seen_frame = frame_num
            current_lost_too_long = False
        elif not faces and current_lost_too_long:
            state.current_cx = default_cx
            state.current_area = 0.0
            state.reset_pending()

        other_faces = [f for f in faces if abs(f.cx - state.current_cx) > match_distance_px]
        candidate = pick_best_face(other_faces, frame_asd) if other_faces else (
            best_face if best_face is not None and abs(best_face.cx - state.current_cx) > match_distance_px else None
        )

        can_switch = not state.is_locked(frame_num)
        if candidate is not None and can_switch:
            size_wins = state.current_area <= 0 or candidate.area >= state.current_area * config.FOCUS_SWITCH_AREA_RATIO
            candidate_is_better = current_lost_too_long or size_wins

            if candidate_is_better:
                if state.pending_cx is None or abs(candidate.cx - state.pending_cx) > match_distance_px:
                    state.pending_cx = candidate.cx
                    state.pending_since_frame = frame_num
                else:
                    state.pending_cx = candidate.cx

                if state.pending_since_frame is None:
                    state.pending_since_frame = frame_num
                pending_age = frame_num - state.pending_since_frame
                if pending_age >= confirm_frames:
                    if abs(candidate.cx - state.current_cx) > match_distance_px * 0.5:
                        smooth_focus_changes_ref[0] += 1
                    state.current_cx = candidate.cx
                    state.current_area = candidate.area
                    state.last_seen_frame = frame_num
                    state.lock(frame_num, min_lock_frames)
            elif current_visible:
                state.reset_pending()
        elif current_visible:
            state.reset_pending()

        add_keyframe(frame_num, state.current_cx)
        state.prev_sample_frame = frame_num

    if not keyframes:
        add_keyframe(0, default_cx)

    raw_targets, hcf = _interpolate_targets_by_scene(keyframes, scene_cut_frames, total_frames, default_cx)
    hcf |= hard_cut_frame_set
    centers, smooth_stats = _apply_crop_smoothing(raw_targets, scene_cut_frames, crop_w, total_frames, src_fps, hard_cut_frames=hcf)

    stats = {
        "scene_cuts": len(scene_cut_frames),
        "target_keyframes": len(keyframes),
        "source_cut_resets": source_cut_resets,
        "smooth_focus_changes": smooth_focus_changes_ref[0],
        **smooth_stats,
    }
    return np.clip(centers, clamp_min, clamp_max), stats


def compute_crop_centers_streaming(clip_path, face_model, src_w, src_h, src_fps,
                                   asd_scores: dict | None = None):
    """Single-pass: scene cuts + face sampling + keyframe building in one video decode."""
    src_fps = max(src_fps, 1.0)

    crop_w = int(src_h * 9 / 16)
    half_crop = crop_w / 2
    default_cx = src_w / 2
    clamp_min = half_crop
    clamp_max = src_w - half_crop

    frame_interval = max(1, int(src_fps / config.FACE_SAMPLE_FPS))
    min_gap_frames = max(1, int(src_fps * config.SCENE_CUT_MIN_GAP_SEC))
    min_lock_frames   = max(1, int(src_fps * config.FOCUS_MIN_LOCK_SEC))
    confirm_frames    = int(src_fps * config.FOCUS_SWITCH_CONFIRM_SEC)
    lost_grace_frames = int(src_fps * config.FOCUS_LOST_GRACE_SEC)
    match_distance_px = max(crop_w * config.FOCUS_MATCH_DISTANCE_RATIO, config.CROP_MIN_DEADZONE_PX)

    scene_state = _SceneCutState(min_gap_frames)
    scene_cut_frames: list[int] = []

    focus_state = FocusTrackerState()
    keyframes: list = []
    hard_cut_frame_set: set = set()
    smooth_focus_changes_ref = [0]
    source_cut_resets = 0

    cap = cv2.VideoCapture(clip_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_scene_cut = _detect_scene_cut(frame, scene_state, frame_idx, min_gap_frames)
        if is_scene_cut:
            scene_cut_frames.append(frame_idx)
            # Reset pending focus state on any scene cut (CR3)
            focus_state.reset_pending()
            focus_state.lock_until_frame = -1

        if frame_idx % frame_interval == 0:
            faces = sample_face_frame(frame, face_model)
            _update_focus(
                state=focus_state,
                faces=faces,
                frame_idx=frame_idx,
                is_scene_cut=is_scene_cut,
                default_cx=default_cx,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                min_lock_frames=min_lock_frames,
                confirm_frames=confirm_frames,
                lost_grace_frames=lost_grace_frames,
                match_distance_px=match_distance_px,
                asd_scores=asd_scores,
                keyframes=keyframes,
                hard_cut_frame_set=hard_cut_frame_set,
                smooth_focus_changes_ref=smooth_focus_changes_ref,
            )
            if is_scene_cut and frame_idx > focus_state.prev_sample_frame - 1:
                source_cut_resets += 1

        frame_idx += 1

    cap.release()
    actual_frames = frame_idx

    if not keyframes:
        keyframes.append((0, float(np.clip(default_cx, clamp_min, clamp_max)), False))

    clamped_keyframes = [
        (max(0, min(fn, actual_frames - 1)), cx, hard)
        for fn, cx, hard in keyframes
    ]

    raw_targets, hcf = _interpolate_targets_by_scene(clamped_keyframes, scene_cut_frames, actual_frames, default_cx)
    hcf |= hard_cut_frame_set
    centers, smooth_stats = _apply_crop_smoothing(raw_targets, scene_cut_frames, crop_w, actual_frames, src_fps, hard_cut_frames=hcf)

    stats = {
        "scene_cuts": len(scene_cut_frames),
        "target_keyframes": len(keyframes),
        "source_cut_resets": source_cut_resets,
        "smooth_focus_changes": smooth_focus_changes_ref[0],
        **smooth_stats,
    }
    return np.clip(centers, clamp_min, clamp_max), stats


def compute_dual_crop_centers_streaming(clip_path, face_model, src_w, src_h, src_fps,
                                        asd_scores: dict | None = None):
    """
    Split-screen variant: track two faces simultaneously (left slot and right slot).

    Faces are assigned to slots by their horizontal position relative to src_w/2.
    Each slot is independently smoothed. If only one face is visible, uses
    centers_single for a full-width crop instead of mirroring.

    Returns (centers_left, centers_right, centers_single, is_split, stats).
    """
    src_fps = max(src_fps, 1.0)

    panel_w = int(src_h * 9 / 16) // 2
    half_p  = panel_w / 2

    clamp_left_min  = half_p
    clamp_left_max  = src_w / 2
    clamp_right_min = src_w / 2
    clamp_right_max = src_w - half_p

    default_left  = src_w * 0.25
    default_right = src_w * 0.75

    frame_interval = max(1, int(src_fps / config.FACE_SAMPLE_FPS))
    deadzone_px    = max(panel_w * config.CROP_DEADZONE_RATIO, config.CROP_MIN_DEADZONE_PX)
    max_step_px    = max(1.0, config.CROP_MAX_SPEED_PX_PER_SEC / src_fps)
    alpha          = 1.0 - np.exp(-1.0 / max(src_fps * config.CROP_SMOOTHING_TAU_SEC, 1.0))

    cx_left   = default_left
    cx_right  = default_right
    cx_single = src_w / 2

    crop_w_single = panel_w * 2
    clamp_single_min = crop_w_single / 2
    clamp_single_max = src_w - crop_w_single / 2

    raw_left:     list[tuple[int, float]] = []
    raw_right:    list[tuple[int, float]] = []
    raw_single:   list[tuple[int, float]] = []
    raw_is_split: list[tuple[int, bool]]  = []

    cap = cv2.VideoCapture(clip_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            faces = sample_face_frame(frame, face_model)
            frame_asd = asd_scores.get(frame_idx) if asd_scores else None

            left_faces  = [f for f in faces if f.cx <  src_w / 2]
            right_faces = [f for f in faces if f.cx >= src_w / 2]

            best_left  = pick_best_face(left_faces,  frame_asd)
            best_right = pick_best_face(right_faces, frame_asd)

            two_faces = best_left is not None and best_right is not None

            if not two_faces:
                single_face = best_left or best_right
                if single_face is not None:
                    cx_single = float(np.clip(single_face.cx, clamp_single_min, clamp_single_max))
                raw_left.append((frame_idx, cx_left))
                raw_right.append((frame_idx, cx_right))
                raw_single.append((frame_idx, cx_single))
                raw_is_split.append((frame_idx, False))
                frame_idx += 1
                continue

            cx_left  = float(np.clip(best_left.cx,  clamp_left_min,  clamp_left_max))
            cx_right = float(np.clip(best_right.cx, clamp_right_min, clamp_right_max))

            raw_left.append((frame_idx, cx_left))
            raw_right.append((frame_idx, cx_right))
            raw_single.append((frame_idx, cx_single))
            raw_is_split.append((frame_idx, True))

        frame_idx += 1

    cap.release()
    actual_frames = frame_idx

    if not raw_left:
        raw_left   = [(0, default_left)]
        raw_right  = [(0, default_right)]
        raw_single = [(0, src_w / 2)]
        raw_is_split = [(0, False)]

    def _interpolate(keyframes, default, total):
        arr = np.full(total, default, dtype=float)
        for i in range(len(keyframes) - 1):
            fa, ca = keyframes[i]
            fb, cb = keyframes[i + 1]
            span = fb - fa
            if span <= 0:
                continue
            for fn in range(fa, min(fb + 1, total)):
                t = (fn - fa) / span
                t = t * t * (3 - 2 * t)
                arr[fn] = ca + (cb - ca) * t
        if keyframes:
            arr[keyframes[-1][0]:] = keyframes[-1][1]
        return arr

    def _smooth(arr, total):
        out = np.empty(total, dtype=float)
        out[0] = arr[0]
        for i in range(1, total):
            delta = arr[i] - out[i - 1]
            if abs(delta) <= deadzone_px:
                out[i] = out[i - 1]
            else:
                desired = arr[i] - np.sign(delta) * deadzone_px
                step = float(np.clip((desired - out[i - 1]) * alpha, -max_step_px, max_step_px))
                out[i] = out[i - 1] + step
        return out

    interp_left   = _interpolate(raw_left,   default_left,  actual_frames)
    interp_right  = _interpolate(raw_right,  default_right, actual_frames)
    interp_single = _interpolate(raw_single, src_w / 2,     actual_frames)

    centers_left   = np.clip(_smooth(interp_left,   actual_frames), clamp_left_min,   clamp_left_max)
    centers_right  = np.clip(_smooth(interp_right,  actual_frames), clamp_right_min,  clamp_right_max)
    centers_single = np.clip(_smooth(interp_single, actual_frames), clamp_single_min, clamp_single_max)

    is_split = np.zeros(actual_frames, dtype=bool)
    for i, (fn, val) in enumerate(raw_is_split):
        next_fn = raw_is_split[i + 1][0] if i + 1 < len(raw_is_split) else actual_frames
        is_split[fn:next_fn] = val

    stats = {
        "scene_cuts": 0,
        "target_keyframes": len(raw_left),
        "source_cut_resets": 0,
        "smooth_focus_changes": 0,
        "hard_crop_jumps": 0,
    }
    return centers_left, centers_right, centers_single, is_split, stats
