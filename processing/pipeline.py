import gc
import os
import re
import json
import cv2
from faster_whisper import WhisperModel
from ultralytics import YOLO

from . import config
from .ffmpeg_utils import cut_clip, extract_audio, composite, composite_split
from .subtitle import write_ass
from .reframe import compute_crop_centers_streaming, compute_dual_crop_centers_streaming
from .asd import load_asd_model, compute_asd_scores
from .face import sample_face_frame
from .logger import get_logger

log = get_logger("processing.pipeline")


def _build_face_tracks(clip_path: str, face_model, src_fps: float) -> list[dict]:
    """Single-pass to collect face detections with boxes for ASD input."""
    from . import config as cfg
    cap = cv2.VideoCapture(clip_path)
    frame_interval = max(1, int(src_fps / cfg.FACE_SAMPLE_FPS))
    tracks = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            faces = sample_face_frame(frame, face_model)
            for face in faces:
                tracks.append({"frame": frame_idx, "box": face.box, "cx": face.cx})
        frame_idx += 1
    cap.release()
    return tracks


def process_clip(clip: dict, whisper_model, face_model, asd_model=None, asd_device=None,
                 mode: str = "single") -> None:
    clip_id     = clip["clip_id"]
    start       = clip["start_time"]
    duration    = str(clip["duration_seconds"])
    raw_caption = clip["suggested_caption"]

    clean_name  = re.sub(r"[^\w\s-]", "", raw_caption)
    clean_name  = re.sub(r"[-\s]+", "_", clean_name).strip("_")
    base_name   = clean_name[:100]

    output_video = os.path.join(config.out_dir, f"{base_name}.mp4")
    temp_clip    = os.path.join(config.out_dir, f"_temp_clip_{clip_id}.mp4")
    temp_audio   = os.path.join(config.out_dir, f"_temp_audio_{clip_id}.wav")
    temp_ass     = os.path.join(config.out_dir, f"_temp_{clip_id}.ass")

    try:
        log.info("[%s] ✂️  Memotong clip...", clip_id)
        cut_clip(start, duration, config.video_file, temp_clip)

        log.info("[%s] 🎧 Transkripsi audio (medium)...", clip_id)
        extract_audio(temp_clip, temp_audio)
        segments, _ = whisper_model.transcribe(temp_audio, language="id", word_timestamps=True)
        words = [w for seg in segments if seg.words for w in seg.words]
        log.debug("[%s] Transkripsi selesai: %d kata", clip_id, len(words))

        log.info("[%s] 🎯 Face detection + crop path...", clip_id)
        cap = cv2.VideoCapture(temp_clip)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps and raw_fps > 0 else 30.0
        cap.release()

        asd_scores = None
        if asd_model is not None and asd_device is not None:
            log.info("[%s] 🗣️  Active speaker detection (Light-ASD)...", clip_id)
            try:
                face_tracks = _build_face_tracks(temp_clip, face_model, fps)
                log.debug("[%s] ASD: %d face track samples dikumpulkan", clip_id, len(face_tracks))
                raw_scores = compute_asd_scores(
                    temp_clip, temp_audio, face_tracks, asd_model, asd_device, fps
                )
                asd_scores = {}
                for track in face_tracks:
                    fn = track["frame"]
                    score = raw_scores.get(fn, 0.0)
                    asd_scores.setdefault(fn, {})[round(track["cx"])] = score
                log.debug("[%s] ASD selesai: %d frame di-score", clip_id, len(asd_scores))
            except Exception:
                log.exception("[%s] ASD gagal, melanjutkan tanpa speaker detection", clip_id)
                asd_scores = None

        if mode == "split":
            log.info("[%s] Mode split screen — tracking dua speaker...", clip_id)
            centers_left, centers_right, centers_single, is_split, crop_stats = compute_dual_crop_centers_streaming(
                temp_clip, face_model, src_w, src_h, fps, asd_scores=asd_scores,
            )
            bottom_margin = int(src_h * (1.0 - config.SPLIT_SCREEN_TOP_RATIO))
            log.info("[%s] 📝 Membuat subtitle ASS (margin bawah %dpx)...", clip_id, bottom_margin)
            write_ass(words, temp_ass, bottom_margin_px=bottom_margin)
            del words
            log.info(
                "[%s] Crop stats — keyframes: %d",
                clip_id, crop_stats["target_keyframes"],
            )
            log.info("[%s] 🎞️  Rendering split screen...", clip_id)
            composite_split(temp_clip, centers_left, centers_right, centers_single, is_split, src_w, src_h, temp_ass, output_video)
            del centers_left, centers_right
        else:
            log.info("[%s] 📝 Membuat subtitle ASS...", clip_id)
            write_ass(words, temp_ass)
            del words
            centers, crop_stats = compute_crop_centers_streaming(
                temp_clip, face_model, src_w, src_h, fps, asd_scores=asd_scores,
            )
            log.info(
                "[%s] Crop stats — scene cuts: %d | hard jumps: %d | focus changes: %d | cut resets: %d",
                clip_id,
                crop_stats["scene_cuts"],
                crop_stats["hard_crop_jumps"],
                crop_stats["smooth_focus_changes"],
                crop_stats["source_cut_resets"],
            )
            log.info("[%s] 🎞️  Rendering final video...", clip_id)
            composite(temp_clip, centers, src_w, src_h, temp_ass, output_video)
            del centers

    except Exception:
        log.exception("[%s] Pipeline gagal", clip_id)
        raise
    finally:
        for path in [temp_clip, temp_audio, temp_ass]:
            if os.path.exists(path):
                os.remove(path)
        gc.collect()

    log.info("[%s] ✅ Selesai! → %s", clip_id, output_video)


def run() -> None:
    log.info("Memuat model Faster-Whisper (medium) di CPU...")
    whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

    log.info("Memuat model YOLOv8 face detection...")
    face_model = YOLO(os.path.join(config.APP_DIR, "yolov8n-face-lindevs.pt"))

    log.info("Memuat model Light-ASD (active speaker detection)...")
    try:
        asd_model, asd_device = load_asd_model()
    except Exception:
        log.exception("Gagal memuat Light-ASD, ASD dinonaktifkan")
        asd_model, asd_device = None, None

    with open(config.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    mode = os.environ.get("AUTOCLIPPER_MODE", "single")
    log.info("Mode: %s", mode)
    log.info("Ditemukan %d clip. Memulai pipeline...", len(data["clips"]))

    for clip in data["clips"]:
        process_clip(clip, whisper_model, face_model, asd_model, asd_device, mode=mode)

    log.info("🎉 Semua pipeline selesai dijalankan!")
