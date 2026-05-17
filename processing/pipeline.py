import gc
import os
import re
import json
import cv2
from faster_whisper import WhisperModel
from ultralytics import YOLO

from . import config
from .ffmpeg_utils import cut_clip, extract_audio, composite
from .subtitle import write_ass
from .reframe import compute_crop_centers_streaming


def process_clip(clip: dict, whisper_model, face_model) -> None:
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

    print(f"[{clip_id}] ✂️  Memotong clip...")
    cut_clip(start, duration, config.video_file, temp_clip)

    print(f"[{clip_id}] 🎧 Transkripsi audio (medium)...")
    extract_audio(temp_clip, temp_audio)
    segments, _ = whisper_model.transcribe(temp_audio, language="id", word_timestamps=True)
    words = [w for seg in segments for w in seg.words]

    print(f"[{clip_id}] 📝 Membuat subtitle ASS...")
    write_ass(words, temp_ass)
    del words

    print(f"[{clip_id}] 🎯 Face detection + scene cuts + crop path (single pass)...")
    cap = cv2.VideoCapture(temp_clip)
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    centers, crop_stats = compute_crop_centers_streaming(
        temp_clip, face_model, src_w, src_h, total_frames, fps,
    )
    print(f"[{clip_id}]    Scene cuts: {crop_stats['scene_cuts']}")
    print(f"[{clip_id}]    Hard crop jumps: {crop_stats['hard_crop_jumps']}")
    print(f"[{clip_id}]    Smooth focus changes: {crop_stats['smooth_focus_changes']}")
    print(f"[{clip_id}]    Source cut resets: {crop_stats['source_cut_resets']}")

    print(f"[{clip_id}] 🎞️  Rendering final video...")
    composite(temp_clip, centers, src_w, src_h, temp_ass, output_video)
    del centers

    for path in [temp_clip, temp_audio, temp_ass]:
        if os.path.exists(path):
            os.remove(path)

    gc.collect()
    print(f"[{clip_id}] ✅ Selesai! → {output_video}")


def run() -> None:
    print("Memuat model Faster-Whisper (medium) di CPU...")
    whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

    print("Memuat model YOLOv8 face detection...")
    face_model = YOLO(os.path.join(config.APP_DIR, "yolov8n-face-lindevs.pt"))

    with open(config.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nDitemukan {len(data['clips'])} clip. Memulai pipeline...\n")

    for clip in data["clips"]:
        process_clip(clip, whisper_model, face_model)

    print("\n🎉 Semua pipeline selesai dijalankan!")
