import cv2
import numpy as np
from . import config
from .face import _safe_fps, pick_best_face, match_face_by_center, sample_face_frame


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


def compute_crop_centers(face_data, scene_cut_frames, src_w, src_h, total_frames, src_fps,
                         asd_scores: dict | None = None):
    """Legacy entry point used by tests or external callers."""
    # face_data format: list of {"frame": int, "faces": [...]}
    # asd_scores: optional dict of frame_num → {cx_key → speaking_prob} from Light-ASD
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
    prev_sample_frame = -1

    current_cx = None
    current_area = 0.0
    last_seen_frame = -(10**9)
    lock_until_frame = -1
    pending_cx = None
    pending_since_frame = None

    keyframes = []
    smooth_focus_changes = 0
    source_cut_resets = 0
    hard_cut_frame_set = set()

    def clamp_cx(cx):
        return float(np.clip(cx, clamp_min, clamp_max))

    def add_keyframe(fn, cx, hard=False):
        if total_frames <= 0:
            return
        fn = max(0, min(int(fn), total_frames - 1))
        keyframes.append((fn, clamp_cx(cx), hard))
        if hard:
            hard_cut_frame_set.add(fn)

    for sample_idx, sample in enumerate(face_data):
        frame_num = int(sample["frame"])
        faces = sample["faces"]
        frame_asd = asd_scores.get(frame_num) if asd_scores else None
        best_face = pick_best_face(faces, frame_asd)

        source_cut_reset = False
        while cut_idx < len(scene_cuts) and scene_cuts[cut_idx] <= frame_num:
            if scene_cuts[cut_idx] > prev_sample_frame:
                source_cut_reset = True
            cut_idx += 1

        if source_cut_reset:
            source_cut_resets += 1
            pending_cx = None
            pending_since_frame = None
            current_cx = best_face["cx"] if best_face is not None else default_cx
            current_area = best_face["area"] if best_face is not None else 0.0
            last_seen_frame = frame_num if best_face is not None else -(10**9)
            lock_until_frame = frame_num + min_lock_frames
            add_keyframe(frame_num, current_cx)
            prev_sample_frame = frame_num
            continue

        if current_cx is None:
            current_cx = best_face["cx"] if best_face is not None else default_cx
            current_area = best_face["area"] if best_face is not None else 0.0
            if best_face is not None:
                last_seen_frame = frame_num
            lock_until_frame = frame_num + min_lock_frames
            add_keyframe(frame_num, current_cx)
            prev_sample_frame = frame_num
            continue

        current_face = match_face_by_center(faces, current_cx, match_distance_px)
        current_visible = current_face is not None
        current_lost_too_long = frame_num - last_seen_frame > lost_grace_frames

        if current_visible:
            current_cx = current_face["cx"]
            current_area = current_face["area"]
            last_seen_frame = frame_num
            current_lost_too_long = False
        elif not faces and current_lost_too_long:
            current_cx = default_cx
            current_area = 0.0
            pending_cx = None
            pending_since_frame = None

        candidate = None
        other_faces = [f for f in faces if abs(f["cx"] - current_cx) > match_distance_px]
        if other_faces:
            candidate = pick_best_face(other_faces, frame_asd)
        elif best_face is not None and abs(best_face["cx"] - current_cx) > match_distance_px:
            candidate = best_face

        can_switch = frame_num >= lock_until_frame
        if candidate is not None and can_switch:
            size_wins = current_area <= 0 or candidate["area"] >= current_area * config.FOCUS_SWITCH_AREA_RATIO
            candidate_is_better = current_lost_too_long or size_wins

            if candidate_is_better:
                if pending_cx is None or abs(candidate["cx"] - pending_cx) > match_distance_px:
                    pending_cx = candidate["cx"]
                    pending_since_frame = frame_num
                else:
                    pending_cx = candidate["cx"]

                if pending_since_frame is None:
                    pending_since_frame = frame_num
                pending_age = frame_num - pending_since_frame
                if pending_age >= confirm_frames:
                    if abs(candidate["cx"] - current_cx) > match_distance_px * 0.5:
                        smooth_focus_changes += 1
                    current_cx = candidate["cx"]
                    current_area = candidate["area"]
                    last_seen_frame = frame_num
                    lock_until_frame = frame_num + min_lock_frames
                    pending_cx = None
                    pending_since_frame = None
            elif current_visible:
                pending_cx = None
                pending_since_frame = None
        elif current_visible:
            pending_cx = None
            pending_since_frame = None

        add_keyframe(frame_num, current_cx)
        prev_sample_frame = frame_num

    if not keyframes:
        add_keyframe(0, default_cx)

    raw_targets, hcf = _interpolate_targets_by_scene(keyframes, scene_cut_frames, total_frames, default_cx)
    hcf |= hard_cut_frame_set
    centers, smooth_stats = _apply_crop_smoothing(raw_targets, scene_cut_frames, crop_w, total_frames, src_fps, hard_cut_frames=hcf)

    stats = {
        "scene_cuts": len(scene_cut_frames),
        "target_keyframes": len(keyframes),
        "source_cut_resets": source_cut_resets,
        "smooth_focus_changes": smooth_focus_changes,
        **smooth_stats,
    }
    return np.clip(centers, clamp_min, clamp_max), stats


def compute_crop_centers_streaming(clip_path, face_model, src_w, src_h, src_fps,
                                   asd_scores: dict | None = None):
    """Single-pass: scene cuts + face sampling + keyframe building in one video decode."""
    # CR4: clamp fps to ≥1 so frame counts are never 0 (B2 fix also applied here)
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

    # scene cut state
    prev_gray = None
    prev_hist = None
    last_cut_frame = -min_gap_frames
    scene_cut_frames = []

    # focus tracking state
    current_cx = None
    current_area = 0.0
    last_seen_frame = -(10**9)
    lock_until_frame = -1
    pending_cx = None
    pending_since_frame = None
    prev_sample_frame = -1

    keyframes = []
    hard_cut_frame_set = set()
    smooth_focus_changes = 0
    source_cut_resets = 0

    def clamp_cx(cx):
        return float(np.clip(cx, clamp_min, clamp_max))

    def add_keyframe(fn, cx, hard=False):
        fn = int(fn)
        keyframes.append((fn, clamp_cx(cx), hard))
        if hard:
            hard_cut_frame_set.add(fn)

    cap = cv2.VideoCapture(clip_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- scene cut detection (every frame, cheap 160x90) ---
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist  = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        is_scene_cut = False
        if prev_gray is not None and prev_hist is not None:
            pixel_diff = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
            hist_diff  = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            score = 0.60 * hist_diff + 0.40 * pixel_diff

            normal_cut   = score >= config.SCENE_CUT_SCORE_THRESHOLD and hist_diff >= config.SCENE_CUT_HIST_THRESHOLD and pixel_diff >= config.SCENE_CUT_PIXEL_THRESHOLD
            strong_hist  = hist_diff >= max(0.45, config.SCENE_CUT_HIST_THRESHOLD * 2.5) and score >= config.SCENE_CUT_SCORE_THRESHOLD
            strong_pixel = pixel_diff >= max(0.22, config.SCENE_CUT_PIXEL_THRESHOLD * 2.0) and hist_diff >= config.SCENE_CUT_HIST_THRESHOLD * 0.75

            if (normal_cut or strong_hist or strong_pixel) and frame_idx - last_cut_frame >= min_gap_frames:
                scene_cut_frames.append(frame_idx)
                last_cut_frame = frame_idx
                is_scene_cut = True

        # CR2: update prev_gray/prev_hist AFTER scene cut detection, before any continue,
        # so the next frame always compares against the most recent frame (not one before the cut).
        prev_gray = gray
        prev_hist = hist

        # CR3: reset pending focus state on ANY scene cut, not just sampling frames.
        if is_scene_cut:
            pending_cx = None
            pending_since_frame = None
            lock_until_frame = -1

        # --- face sampling (every frame_interval frames) ---
        if frame_idx % frame_interval == 0:
            faces = sample_face_frame(frame, face_model)
            frame_asd = asd_scores.get(frame_idx) if asd_scores else None
            best_face = pick_best_face(faces, frame_asd)

            # handle scene cut reset at a sampling frame
            if is_scene_cut and frame_idx > prev_sample_frame:
                source_cut_resets += 1
                # pending_cx/lock already reset above (CR3)
                current_cx = best_face["cx"] if best_face is not None else default_cx
                current_area = best_face["area"] if best_face is not None else 0.0
                last_seen_frame = frame_idx if best_face is not None else -(10**9)
                lock_until_frame = frame_idx + min_lock_frames
                add_keyframe(frame_idx, current_cx)
                prev_sample_frame = frame_idx
                # C3: NO frame_idx += 1 here; the single increment is at the bottom of the loop.
            elif current_cx is None:
                current_cx = best_face["cx"] if best_face is not None else default_cx
                current_area = best_face["area"] if best_face is not None else 0.0
                if best_face is not None:
                    last_seen_frame = frame_idx
                lock_until_frame = frame_idx + min_lock_frames
                add_keyframe(frame_idx, current_cx)
                prev_sample_frame = frame_idx
                # C3: NO frame_idx += 1 here.
            else:
                current_face = match_face_by_center(faces, current_cx, match_distance_px)
                current_visible = current_face is not None
                current_lost_too_long = frame_idx - last_seen_frame > lost_grace_frames

                if current_visible:
                    current_cx = current_face["cx"]
                    current_area = current_face["area"]
                    last_seen_frame = frame_idx
                    current_lost_too_long = False
                elif not faces and current_lost_too_long:
                    current_cx = default_cx
                    current_area = 0.0
                    pending_cx = None
                    pending_since_frame = None

                candidate = None
                other_faces = [f for f in faces if abs(f["cx"] - current_cx) > match_distance_px]
                if other_faces:
                    candidate = pick_best_face(other_faces, frame_asd)
                elif best_face is not None and abs(best_face["cx"] - current_cx) > match_distance_px:
                    candidate = best_face

                can_switch = frame_idx >= lock_until_frame
                if candidate is not None and can_switch:
                    size_wins = current_area <= 0 or candidate["area"] >= current_area * config.FOCUS_SWITCH_AREA_RATIO
                    candidate_is_better = current_lost_too_long or size_wins

                    if candidate_is_better:
                        if pending_cx is None or abs(candidate["cx"] - pending_cx) > match_distance_px:
                            # B1: new candidate position — reset timer so it counts from THIS frame.
                            pending_cx = candidate["cx"]
                            pending_since_frame = frame_idx
                        else:
                            # Same candidate, just update position; keep original pending_since_frame.
                            pending_cx = candidate["cx"]

                        if pending_since_frame is None:
                            pending_since_frame = frame_idx
                        pending_age = frame_idx - pending_since_frame
                        if pending_age >= confirm_frames:
                            if abs(candidate["cx"] - current_cx) > match_distance_px * 0.5:
                                smooth_focus_changes += 1
                            current_cx = candidate["cx"]
                            current_area = candidate["area"]
                            last_seen_frame = frame_idx
                            # M4: clamp lock so it doesn't run past end of video.
                            lock_until_frame = frame_idx + min_lock_frames
                            pending_cx = None
                            pending_since_frame = None
                    elif current_visible:
                        pending_cx = None
                        pending_since_frame = None
                elif current_visible:
                    pending_cx = None
                    pending_since_frame = None

                add_keyframe(frame_idx, current_cx)
                prev_sample_frame = frame_idx

        # C3: single increment — always at the bottom, never inside the sampling block.
        frame_idx += 1

    cap.release()

    # CR4: use actual frame_idx (frames read from cap) as the true total, not CAP_PROP_FRAME_COUNT.
    actual_frames = frame_idx

    if not keyframes:
        add_keyframe(0, default_cx)

    # Clamp all keyframe indices to actual range now that we know it.
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
        "smooth_focus_changes": smooth_focus_changes,
        **smooth_stats,
    }
    return np.clip(centers, clamp_min, clamp_max), stats


def compute_dual_crop_centers_streaming(clip_path, face_model, src_w, src_h, src_fps,
                                        asd_scores: dict | None = None):
    """
    Split-screen variant: track two faces simultaneously (left slot and right slot).

    Faces are assigned to slots by their horizontal position relative to src_w/2.
    Each slot is independently smoothed. If only one face is visible, both slots
    mirror it so the frame is always filled.

    Returns (centers_left, centers_right, stats) where each centers_* is a 1-D
    numpy array of length == actual_frames, containing the crop-center x-coordinate
    within the source frame for that panel.
    """
    src_fps = max(src_fps, 1.0)

    panel_w = int(src_h * 9 / 16) // 2   # half of the 9:16 crop width
    half_p  = panel_w / 2

    # Clamp bounds so panel never goes outside source frame
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

    cx_left  = default_left
    cx_right = default_right

    raw_left:  list[tuple[int, float]] = []
    raw_right: list[tuple[int, float]] = []

    cap = cv2.VideoCapture(clip_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            faces = sample_face_frame(frame, face_model)
            frame_asd = asd_scores.get(frame_idx) if asd_scores else None

            left_faces  = [f for f in faces if f["cx"] <  src_w / 2]
            right_faces = [f for f in faces if f["cx"] >= src_w / 2]

            best_left  = pick_best_face(left_faces,  frame_asd)
            best_right = pick_best_face(right_faces, frame_asd)

            # Mirror if one side is empty
            if best_left is None and best_right is not None:
                best_left = best_right
            elif best_right is None and best_left is not None:
                best_right = best_left
            elif best_left is None and best_right is None:
                raw_left.append((frame_idx, cx_left))
                raw_right.append((frame_idx, cx_right))
                frame_idx += 1
                continue

            cx_left  = float(np.clip(best_left["cx"],  clamp_left_min,  clamp_left_max))
            cx_right = float(np.clip(best_right["cx"], clamp_right_min, clamp_right_max))

            raw_left.append((frame_idx, cx_left))
            raw_right.append((frame_idx, cx_right))

        frame_idx += 1

    cap.release()
    actual_frames = frame_idx

    if not raw_left:
        raw_left  = [(0, default_left)]
        raw_right = [(0, default_right)]

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

    interp_left  = _interpolate(raw_left,  default_left,  actual_frames)
    interp_right = _interpolate(raw_right, default_right, actual_frames)

    centers_left  = np.clip(_smooth(interp_left,  actual_frames), clamp_left_min,  clamp_left_max)
    centers_right = np.clip(_smooth(interp_right, actual_frames), clamp_right_min, clamp_right_max)

    stats = {
        "scene_cuts": 0,
        "target_keyframes": len(raw_left),
        "source_cut_resets": 0,
        "smooth_focus_changes": 0,
        "hard_crop_jumps": 0,
    }
    return centers_left, centers_right, stats
