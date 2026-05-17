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


def compute_crop_centers(face_data, scene_cut_frames, src_w, src_h, total_frames, src_fps):
    """Legacy entry point used by tests or external callers. Streams internally."""
    # face_data format: list of {"frame": int, "faces": [...]}
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
        best_face = pick_best_face(faces)

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
            candidate = pick_best_face(other_faces)
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


def compute_crop_centers_streaming(clip_path, face_model, src_w, src_h, total_frames, src_fps):
    """Single-pass: scene cuts + face sampling + keyframe building in one video decode."""
    crop_w = int(src_h * 9 / 16)
    half_crop = crop_w / 2
    default_cx = src_w / 2
    clamp_min = half_crop
    clamp_max = src_w - half_crop

    frame_interval = max(1, int(src_fps / config.FACE_SAMPLE_FPS))
    min_gap_frames = max(1, int(src_fps * config.SCENE_CUT_MIN_GAP_SEC))

    min_lock_frames   = int(src_fps * config.FOCUS_MIN_LOCK_SEC)
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
        if total_frames <= 0:
            return
        fn = max(0, min(int(fn), total_frames - 1))
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

            normal_cut     = score >= config.SCENE_CUT_SCORE_THRESHOLD and hist_diff >= config.SCENE_CUT_HIST_THRESHOLD and pixel_diff >= config.SCENE_CUT_PIXEL_THRESHOLD
            strong_hist    = hist_diff >= max(0.45, config.SCENE_CUT_HIST_THRESHOLD * 2.5) and score >= config.SCENE_CUT_SCORE_THRESHOLD
            strong_pixel   = pixel_diff >= max(0.22, config.SCENE_CUT_PIXEL_THRESHOLD * 2.0) and hist_diff >= config.SCENE_CUT_HIST_THRESHOLD * 0.75

            if (normal_cut or strong_hist or strong_pixel) and frame_idx - last_cut_frame >= min_gap_frames:
                scene_cut_frames.append(frame_idx)
                last_cut_frame = frame_idx
                is_scene_cut = True

        prev_gray = gray
        prev_hist = hist

        # --- face sampling (every frame_interval frames) ---
        if frame_idx % frame_interval == 0:
            faces = sample_face_frame(frame, face_model)
            best_face = pick_best_face(faces)

            # handle scene cut reset
            if is_scene_cut and frame_idx > prev_sample_frame:
                source_cut_resets += 1
                pending_cx = None
                pending_since_frame = None
                current_cx = best_face["cx"] if best_face is not None else default_cx
                current_area = best_face["area"] if best_face is not None else 0.0
                last_seen_frame = frame_idx if best_face is not None else -(10**9)
                lock_until_frame = frame_idx + min_lock_frames
                add_keyframe(frame_idx, current_cx)
                prev_sample_frame = frame_idx
                frame_idx += 1
                continue

            if current_cx is None:
                current_cx = best_face["cx"] if best_face is not None else default_cx
                current_area = best_face["area"] if best_face is not None else 0.0
                if best_face is not None:
                    last_seen_frame = frame_idx
                lock_until_frame = frame_idx + min_lock_frames
                add_keyframe(frame_idx, current_cx)
                prev_sample_frame = frame_idx
                frame_idx += 1
                continue

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
                candidate = pick_best_face(other_faces)
            elif best_face is not None and abs(best_face["cx"] - current_cx) > match_distance_px:
                candidate = best_face

            can_switch = frame_idx >= lock_until_frame
            if candidate is not None and can_switch:
                size_wins = current_area <= 0 or candidate["area"] >= current_area * config.FOCUS_SWITCH_AREA_RATIO
                candidate_is_better = current_lost_too_long or size_wins

                if candidate_is_better:
                    if pending_cx is None or abs(candidate["cx"] - pending_cx) > match_distance_px:
                        pending_cx = candidate["cx"]
                        pending_since_frame = frame_idx
                    else:
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

        frame_idx += 1

    cap.release()

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
