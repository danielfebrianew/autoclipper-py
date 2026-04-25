import os
from dotenv import load_dotenv
load_dotenv()
import json
import subprocess
import re
import cv2
import numpy as np
from faster_whisper import WhisperModel
from ultralytics import YOLO

# ==========================================
# 1. KONFIGURASI FILE & MODEL
# ==========================================
json_file      = os.environ["AUTOCLIPPER_JSON"]
video_file     = os.environ["AUTOCLIPPER_VIDEO"]
out_dir        = os.environ.get("AUTOCLIPPER_OUTDIR",        '.')
channel_name   = os.environ.get("AUTOCLIPPER_CHANNEL",       '')
source_credit  = os.environ.get("AUTOCLIPPER_SOURCE_CREDIT", '')

APP_DIR = os.path.dirname(os.path.abspath(__file__))

print("Memuat model Faster-Whisper (medium) di CPU...")
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

print("Memuat model YOLOv8 face detection...")
face_model = YOLO(os.path.join(APP_DIR, "yolov8n-face-lindevs.pt"))

MAX_WORDS_PER_SCREEN = 2

FACE_SAMPLE_FPS = float(os.environ.get("AUTOCLIPPER_FACE_SAMPLE_FPS", "4"))

# Reframe tuning. Defaults lean stable so the crop does not chase tiny face jitter.
SCENE_CUT_SCORE_THRESHOLD = float(os.environ.get("AUTOCLIPPER_SCENE_CUT_SCORE", "0.22"))
SCENE_CUT_HIST_THRESHOLD = float(os.environ.get("AUTOCLIPPER_SCENE_CUT_HIST", "0.14"))
SCENE_CUT_PIXEL_THRESHOLD = float(os.environ.get("AUTOCLIPPER_SCENE_CUT_PIXEL", "0.08"))
SCENE_CUT_MIN_GAP_SEC = float(os.environ.get("AUTOCLIPPER_SCENE_CUT_MIN_GAP", "0.30"))

FOCUS_MIN_LOCK_SEC = float(os.environ.get("AUTOCLIPPER_FOCUS_MIN_LOCK", "1.50"))
FOCUS_SWITCH_CONFIRM_SEC = float(os.environ.get("AUTOCLIPPER_FOCUS_CONFIRM", "0.85"))
FOCUS_SWITCH_AREA_RATIO = float(os.environ.get("AUTOCLIPPER_FOCUS_AREA_RATIO", "1.35"))
FOCUS_LOST_GRACE_SEC = float(os.environ.get("AUTOCLIPPER_FOCUS_LOST_GRACE", "0.80"))
FOCUS_MATCH_DISTANCE_RATIO = float(os.environ.get("AUTOCLIPPER_FOCUS_MATCH_DISTANCE", "0.35"))

CROP_DEADZONE_RATIO = float(os.environ.get("AUTOCLIPPER_CROP_DEADZONE", "0.07"))
CROP_MIN_DEADZONE_PX = float(os.environ.get("AUTOCLIPPER_CROP_MIN_DEADZONE_PX", "36"))
CROP_SMOOTHING_TAU_SEC = float(os.environ.get("AUTOCLIPPER_CROP_SMOOTHING_TAU", "0.45"))
CROP_MAX_SPEED_PX_PER_SEC = float(os.environ.get("AUTOCLIPPER_CROP_MAX_SPEED", "480"))


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def format_timestamp_ass(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def cut_clip(start, duration, src_video, dest):
    result = subprocess.run([
        'ffmpeg', '-ss', str(start), '-i', src_video,
        '-t', str(duration), '-c', 'copy', '-y', dest,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode('utf-8', errors='replace')[-1000:]
        print(f"[ffmpeg stderr]\n{err}", flush=True)
        raise subprocess.CalledProcessError(result.returncode, 'ffmpeg', stderr=err)


def extract_audio(src_video, dest_wav):
    subprocess.run([
        'ffmpeg', '-i', src_video,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', dest_wav,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def write_ass(segments_list, ass_path):
    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 608
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,64,&H00FFFFFF,&H0000FFFF,&H00000000,&HCC000000,-1,0,0,0,100,100,0,0,1,4,3,2,10,10,450,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # ASS color format: &H00BBGGRR (no alpha for primary)
    WHITE  = "&H00FFFFFF"
    YELLOW = "&H0000FFFF"

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
                    parts.append(f"{{\\c{YELLOW}}}{word_text}{{\\c{WHITE}}}")
                else:
                    parts.append(word_text)
            text = " ".join(parts)
            f.write(f"Dialogue: 0,{format_timestamp_ass(seg_start)},{format_timestamp_ass(seg_end)},Default,,0,0,0,,{text}\n")

    all_words = [w for seg in segments_list for w in seg.words]
    chunks = [all_words[i:i + MAX_WORDS_PER_SCREEN]
              for i in range(0, len(all_words), MAX_WORDS_PER_SCREEN)]

    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_header)
        for chunk in chunks:
            write_chunk(f, chunk)


def _safe_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else 30.0


def detect_faces(clip_path, sample_fps=FACE_SAMPLE_FPS):
    """Returns face samples with all detected face candidates per sampled frame."""
    cap = cv2.VideoCapture(clip_path)
    src_fps = _safe_fps(cap)
    frame_interval = max(1, int(src_fps / sample_fps))
    results = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            t = frame_idx / src_fps
            detections = face_model(frame, verbose=False)[0]
            faces = []
            if len(detections.boxes) > 0:
                boxes = detections.boxes.xyxy.cpu().numpy()
                confs = detections.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box
                    area = max(0.0, (x2 - x1) * (y2 - y1))
                    if area <= 0:
                        continue
                    faces.append({
                        "cx": float((x1 + x2) / 2),
                        "cy": float((y1 + y2) / 2),
                        "area": float(area),
                        "conf": float(conf),
                        "box": (float(x1), float(y1), float(x2), float(y2)),
                    })
                faces.sort(key=lambda face: face["area"], reverse=True)
            results.append({"t": t, "frame": frame_idx, "faces": faces})
        frame_idx += 1

    cap.release()
    return results


def detect_scene_cuts(clip_path):
    """Returns frame numbers where the source video appears to hard-cut."""
    cap = cv2.VideoCapture(clip_path)
    src_fps = _safe_fps(cap)
    min_gap_frames = max(1, int(src_fps * SCENE_CUT_MIN_GAP_SEC))

    cuts = []
    prev_gray = None
    prev_hist = None
    last_cut_frame = -min_gap_frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if prev_gray is not None and prev_hist is not None:
            pixel_diff = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
            hist_diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            score = (0.60 * hist_diff) + (0.40 * pixel_diff)

            normal_cut = (
                score >= SCENE_CUT_SCORE_THRESHOLD
                and hist_diff >= SCENE_CUT_HIST_THRESHOLD
                and pixel_diff >= SCENE_CUT_PIXEL_THRESHOLD
            )
            strong_hist_cut = (
                hist_diff >= max(0.45, SCENE_CUT_HIST_THRESHOLD * 2.5)
                and score >= SCENE_CUT_SCORE_THRESHOLD
            )
            strong_pixel_cut = (
                pixel_diff >= max(0.22, SCENE_CUT_PIXEL_THRESHOLD * 2.0)
                and hist_diff >= SCENE_CUT_HIST_THRESHOLD * 0.75
            )
            looks_like_cut = normal_cut or strong_hist_cut or strong_pixel_cut

            if looks_like_cut and frame_idx - last_cut_frame >= min_gap_frames:
                cuts.append(frame_idx)
                last_cut_frame = frame_idx

        prev_gray = gray
        prev_hist = hist
        frame_idx += 1

    cap.release()
    return cuts


def _face_score(face):
    return face["area"] * (0.75 + face.get("conf", 1.0))


def _pick_best_face(faces):
    if not faces:
        return None
    return max(faces, key=_face_score)


def _match_face_by_center(faces, current_cx, max_distance_px):
    if current_cx is None or not faces:
        return None
    nearest = min(faces, key=lambda face: abs(face["cx"] - current_cx))
    if abs(nearest["cx"] - current_cx) <= max_distance_px:
        return nearest
    return None


def _build_focus_keyframes(face_data, scene_cut_frames, src_w, src_h, total_frames, src_fps):
    crop_w = int(src_h * 9 / 16)
    half_crop = crop_w / 2
    clamp_min = half_crop
    clamp_max = src_w - half_crop
    default_cx = src_w / 2

    min_lock_frames = int(src_fps * FOCUS_MIN_LOCK_SEC)
    confirm_frames = int(src_fps * FOCUS_SWITCH_CONFIRM_SEC)
    lost_grace_frames = int(src_fps * FOCUS_LOST_GRACE_SEC)
    match_distance_px = max(crop_w * FOCUS_MATCH_DISTANCE_RATIO, CROP_MIN_DEADZONE_PX)

    scene_cuts = sorted(fn for fn in scene_cut_frames if 0 < fn < total_frames)
    cut_idx = 0
    prev_sample_frame = -1

    current_cx = None
    current_area = 0.0
    last_seen_frame = -10**9
    lock_until_frame = -1
    pending_cx = None
    pending_since_frame = None

    keyframes = []
    smooth_focus_changes = 0
    source_cut_resets = 0

    def clamp_cx(cx):
        return float(np.clip(cx, clamp_min, clamp_max))

    def add_keyframe(frame_num, cx):
        if total_frames <= 0:
            return
        fn = max(0, min(int(frame_num), total_frames - 1))
        keyframes.append((fn, clamp_cx(cx)))

    for sample in face_data:
        frame_num = int(sample["frame"])
        faces = sample["faces"]
        best_face = _pick_best_face(faces)

        source_cut_reset = False
        while cut_idx < len(scene_cuts) and scene_cuts[cut_idx] <= frame_num:
            if scene_cuts[cut_idx] > prev_sample_frame:
                source_cut_reset = True
            cut_idx += 1

        if source_cut_reset:
            source_cut_resets += 1
            pending_cx = None
            pending_since_frame = None
            if best_face is not None:
                current_cx = best_face["cx"]
                current_area = best_face["area"]
                last_seen_frame = frame_num
            else:
                current_cx = default_cx
                current_area = 0.0
                last_seen_frame = -10**9
            lock_until_frame = frame_num + min_lock_frames
            add_keyframe(frame_num, current_cx)
            prev_sample_frame = frame_num
            continue

        if current_cx is None:
            if best_face is not None:
                current_cx = best_face["cx"]
                current_area = best_face["area"]
                last_seen_frame = frame_num
            else:
                current_cx = default_cx
                current_area = 0.0
            lock_until_frame = frame_num + min_lock_frames
            add_keyframe(frame_num, current_cx)
            prev_sample_frame = frame_num
            continue

        current_face = _match_face_by_center(faces, current_cx, match_distance_px)
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
        if best_face is not None:
            far_from_current = abs(best_face["cx"] - current_cx) > match_distance_px
            if not current_visible or far_from_current:
                candidate = best_face

        can_switch = frame_num >= lock_until_frame
        if candidate is not None and can_switch:
            candidate_is_better = (
                current_lost_too_long
                or current_area <= 0
                or candidate["area"] >= current_area * FOCUS_SWITCH_AREA_RATIO
            )

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

    return keyframes, {
        "source_cut_resets": source_cut_resets,
        "smooth_focus_changes": smooth_focus_changes,
    }


def _interpolate_targets_by_scene(keyframes, scene_cut_frames, total_frames, default_cx):
    if total_frames <= 0:
        return np.array([], dtype=float)

    raw_targets = np.full(total_frames, default_cx, dtype=float)
    deduped = {}
    for frame_num, cx in keyframes:
        if 0 <= frame_num < total_frames:
            deduped[int(frame_num)] = float(cx)

    if not deduped:
        return raw_targets

    sorted_keys = sorted(deduped.items())
    scene_cuts = sorted(set(fn for fn in scene_cut_frames if 0 < fn < total_frames))
    segment_starts = [0] + scene_cuts
    segment_ends = scene_cuts + [total_frames]

    def ease(t):
        return t * t * (3 - 2 * t)

    for seg_start, seg_end in zip(segment_starts, segment_ends):
        seg_keys = [(fn, cx) for fn, cx in sorted_keys if seg_start <= fn < seg_end]
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

    return raw_targets


def _apply_crop_smoothing(raw_targets, scene_cut_frames, crop_w, total_frames, src_fps):
    if total_frames <= 0:
        return raw_targets, {"hard_crop_jumps": 0}

    centers = np.empty(total_frames, dtype=float)
    centers[0] = raw_targets[0]

    cut_frames = set(fn for fn in scene_cut_frames if 0 < fn < total_frames)
    deadzone_px = max(crop_w * CROP_DEADZONE_RATIO, CROP_MIN_DEADZONE_PX)
    max_step_px = max(1.0, CROP_MAX_SPEED_PX_PER_SEC / max(src_fps, 1.0))
    alpha = 1.0 - np.exp(-1.0 / max(src_fps * CROP_SMOOTHING_TAU_SEC, 1.0))
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
        step = (desired_cx - centers[frame_num - 1]) * alpha
        step = float(np.clip(step, -max_step_px, max_step_px))
        centers[frame_num] = centers[frame_num - 1] + step

    return centers, {"hard_crop_jumps": hard_crop_jumps}


def compute_crop_centers(face_data, scene_cut_frames, src_w, src_h, total_frames, src_fps):
    """Returns smoothed center_x values plus debug stats for the crop path."""
    crop_w = int(src_h * 9 / 16)
    half_crop = crop_w / 2
    clamp_min = half_crop
    clamp_max = src_w - half_crop
    default_cx = src_w / 2

    keyframes, focus_stats = _build_focus_keyframes(
        face_data, scene_cut_frames, src_w, src_h, total_frames, src_fps,
    )
    raw_targets = _interpolate_targets_by_scene(
        keyframes, scene_cut_frames, total_frames, default_cx,
    )
    centers, smooth_stats = _apply_crop_smoothing(
        raw_targets, scene_cut_frames, crop_w, total_frames, src_fps,
    )

    stats = {
        "scene_cuts": len(scene_cut_frames),
        "target_keyframes": len(keyframes),
        **focus_stats,
        **smooth_stats,
    }

    return np.clip(centers, clamp_min, clamp_max), stats


def _escape_drawtext(text):
    return text.replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")


def composite(temp_clip, centers, src_w, src_h, ass_path, output_video):
    crop_w = int(src_h * 9 / 16)
    crop_h = src_h

    cap = cv2.VideoCapture(temp_clip)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    safe_ass = ass_path.replace('\\', '/').replace(':', '\\:')
    filters = [f"ass={safe_ass}"]

    if source_credit.strip():
        txt = _escape_drawtext(source_credit.strip())
        filters.append(
            f"drawtext=text='{txt}':font=Impact:fontsize=28"
            f":fontcolor=white@0.55:x=(w-text_w)/2:y=40"
            f":shadowcolor=black@0.6:shadowx=2:shadowy=2"
        )

    if channel_name.strip():
        txt = _escape_drawtext(channel_name.strip())
        filters.insert(1,
            f"drawtext=text='{txt}':font=Impact:fontsize=28"
            f":fontcolor=white@0.20:x=(w-text_w)/2:y=(h-text_h)/2"
            f":shadowcolor=black@0.20:shadowx=2:shadowy=2"
        )

    vf = ",".join(filters)

    # Pipe raw BGR frames ke ffmpeg. OpenCV VideoWriter di macOS M-series
    # sering bikin mp4 yg tidak valid; pakai pipe lebih reliable.
    proc = subprocess.Popen([
        'ffmpeg',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{crop_w}x{crop_h}',
        '-pix_fmt', 'bgr24',
        '-r', f'{src_fps}',
        '-i', '-',                     # video from stdin
        '-i', temp_clip,               # audio source
        '-map', '0:v', '-map', '1:a',
        '-vf', vf,
        '-c:v', 'h264_videotoolbox',
        '-pix_fmt', 'yuv420p',
        '-b:v', '5000k',
        '-c:a', 'aac',
        '-shortest',
        '-y', output_video,
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    assert proc.stdin is not None

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cx = centers[min(frame_idx, len(centers) - 1)]
            x1 = int(cx - crop_w / 2)
            x1 = max(0, min(x1, src_w - crop_w))
            cropped = frame[:, x1:x1 + crop_w]
            proc.stdin.write(cropped.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass

    _, stderr = proc.communicate()
    if proc.returncode != 0:
        err_tail = (stderr or b'').decode('utf-8', errors='replace')[-1500:]
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, output=None, stderr=err_tail,
        )


# ==========================================
# 3. PROSES UTAMA
# ==========================================
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"\nDitemukan {len(data['clips'])} clip. Memulai pipeline...\n")

for clip in data['clips']:
    clip_id  = clip['clip_id']
    start    = clip['start_time']
    duration = str(clip['duration_seconds'])
    raw_caption = clip['suggested_caption']

    clean_name = re.sub(r'[^\w\s-]', '', raw_caption)
    clean_name = re.sub(r'[-\s]+', '_', clean_name).strip('_')
    base_name = clean_name[:100]

    output_video = os.path.join(out_dir, f"{base_name}.mp4")
    temp_clip    = os.path.join(out_dir, f"_temp_clip_{clip_id}.mp4")
    temp_audio   = os.path.join(out_dir, f"_temp_audio_{clip_id}.wav")
    temp_ass     = os.path.join(out_dir, f"_temp_{clip_id}.ass")

    # ── 1. CUT ────────────────────────────────────────────────
    print(f"[{clip_id}] ✂️  Memotong clip...")
    cut_clip(start, duration, video_file, temp_clip)

    # ── 2. TRANSCRIBE ─────────────────────────────────────────
    print(f"[{clip_id}] 🎧 Transkripsi audio (medium)...")
    extract_audio(temp_clip, temp_audio)
    segments, _ = whisper_model.transcribe(temp_audio, language="id", word_timestamps=True)
    segments_list = list(segments)

    # ── 3. SUBTITLE ───────────────────────────────────────────
    print(f"[{clip_id}] 📝 Membuat subtitle ASS...")
    write_ass(segments_list, temp_ass)

    # ── 4. FACE DETECTION ─────────────────────────────────────
    print(f"[{clip_id}] 👤 Deteksi wajah ({FACE_SAMPLE_FPS:g}fps sampling)...")
    face_data = detect_faces(temp_clip, sample_fps=FACE_SAMPLE_FPS)
    samples_with_faces = sum(1 for sample in face_data if sample["faces"])
    total_faces = sum(len(sample["faces"]) for sample in face_data)
    print(f"[{clip_id}]    {samples_with_faces}/{len(face_data)} sample ada wajah ({total_faces} total detections).")

    print(f"[{clip_id}] 🎬 Deteksi scene cut dari video asli...")
    scene_cut_frames = detect_scene_cuts(temp_clip)
    print(f"[{clip_id}]    Scene cuts detected: {len(scene_cut_frames)}")

    # ── 5. REFRAME ────────────────────────────────────────────
    print(f"[{clip_id}] 🎯 Kalkulasi crop path...")
    cap = cv2.VideoCapture(temp_clip)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    centers, crop_stats = compute_crop_centers(
        face_data, scene_cut_frames, src_w, src_h, total_frames, fps,
    )
    print(f"[{clip_id}]    Hard crop jumps used: {crop_stats['hard_crop_jumps']}")
    print(f"[{clip_id}]    Smooth focus changes: {crop_stats['smooth_focus_changes']}")
    print(f"[{clip_id}]    Source cut focus resets: {crop_stats['source_cut_resets']}")

    # ── 6. COMPOSITE ──────────────────────────────────────────
    print(f"[{clip_id}] 🎞️  Rendering final video...")
    composite(temp_clip, centers, src_w, src_h, temp_ass, output_video)

    # ── CLEANUP per-clip ───────────────────────────────────────
    for f in [temp_clip, temp_audio, temp_ass]:
        if os.path.exists(f):
            os.remove(f)

    print(f"[{clip_id}] ✅ Selesai! → {output_video}")

print("\n🎉 Semua pipeline selesai dijalankan!")
