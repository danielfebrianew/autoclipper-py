"""
VideoProcessingService — orchestrates the full clip render pipeline.

Replaces processing/pipeline.py's process_clip() with a class that has
injected dependencies (models, config, ffmpeg) instead of globals.
"""

import gc
import os
import re
import cv2
import numpy as np

from core.models import FocusTrackerState
from core.exceptions import PipelineError
from processing import config
from processing.face import sample_face_frame
from processing.reframe import (
    _SceneCutState,
    _detect_scene_cut,
    _update_focus,
    _interpolate_targets_by_scene,
    _apply_crop_smoothing,
)
from processing.subtitle import write_ass
from processing.asd import compute_asd_scores
from processing.logger import get_logger
from adapters.ffmpeg import FFmpegAdapter

log = get_logger("services.video_processing")


class VideoProcessingService:
    def __init__(
        self,
        whisper_model,
        face_model,
        asd_model=None,
        asd_device=None,
        ffmpeg: FFmpegAdapter | None = None,
    ):
        self.whisper = whisper_model
        self.face_model = face_model
        self.asd_model = asd_model
        self.asd_device = asd_device
        self.ffmpeg = ffmpeg or FFmpegAdapter()

    def process_clip(self, clip: dict, mode: str = "single") -> None:
        clip_id     = clip["clip_id"]
        start       = clip["start_time"]
        duration    = str(clip["duration_seconds"])
        raw_caption = clip["suggested_caption"]

        clean_name  = re.sub(r"[^\w\s-]", "", raw_caption)
        clean_name  = re.sub(r"[-\s]+", "_", clean_name).strip("_")
        output_video = os.path.join(config.out_dir, f"{clean_name[:100]}.mp4")
        temp_clip    = os.path.join(config.out_dir, f"_temp_clip_{clip_id}.mp4")
        temp_audio   = os.path.join(config.out_dir, f"_temp_audio_{clip_id}.wav")
        temp_ass     = os.path.join(config.out_dir, f"_temp_{clip_id}.ass")

        try:
            log.info("[%s] ✂️  Memotong clip...", clip_id)
            self.ffmpeg.cut(start, duration, config.video_file, temp_clip)

            log.info("[%s] 🎧 Transkripsi audio (medium)...", clip_id)
            self.ffmpeg.extract_audio(temp_clip, temp_audio)
            segments, _ = self.whisper.transcribe(temp_audio, language="id", word_timestamps=True)
            words = [w for seg in segments if seg.words for w in seg.words]
            log.debug("[%s] Transkripsi selesai: %d kata", clip_id, len(words))

            log.info("[%s] 🎯 Face detection + crop path...", clip_id)
            cap = cv2.VideoCapture(temp_clip)
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = raw_fps if raw_fps and raw_fps > 0 else 30.0
            cap.release()

            asd_scores = self._run_asd(clip_id, temp_clip, temp_audio, fps)

            if mode == "split":
                self._render_split(clip_id, temp_clip, temp_audio, temp_ass,
                                   words, asd_scores, src_w, src_h, fps, output_video)
            else:
                self._render_single(clip_id, temp_clip, temp_ass,
                                    words, asd_scores, src_w, src_h, fps, output_video)

        except Exception:
            log.exception("[%s] Pipeline gagal", clip_id)
            raise
        finally:
            for path in [temp_clip, temp_audio, temp_ass]:
                if os.path.exists(path):
                    os.remove(path)
            gc.collect()

        log.info("[%s] ✅ Selesai! → %s", clip_id, output_video)

    # ── private ───────────────────────────────────────────────────────────────

    def _run_asd(self, clip_id, temp_clip: str, temp_audio: str, fps: float) -> dict | None:
        if self.asd_model is None:
            return None
        log.info("[%s] 🗣️  Active speaker detection (Light-ASD)...", clip_id)
        try:
            face_tracks = self._build_face_tracks(temp_clip, fps)
            log.debug("[%s] ASD: %d face track samples", clip_id, len(face_tracks))
            raw_scores = compute_asd_scores(
                temp_clip, temp_audio, face_tracks, self.asd_model, self.asd_device, fps
            )
            asd_scores: dict = {}
            for track in face_tracks:
                fn = track["frame"]
                score = raw_scores.get(fn, 0.0)
                asd_scores.setdefault(fn, {})[round(track["cx"])] = score
            log.debug("[%s] ASD selesai: %d frame di-score", clip_id, len(asd_scores))
            return asd_scores
        except Exception:
            log.exception("[%s] ASD gagal, lanjut tanpa speaker detection", clip_id)
            return None

    def _build_face_tracks(self, clip_path: str, src_fps: float) -> list[dict]:
        frame_interval = max(1, int(src_fps / config.FACE_SAMPLE_FPS))
        cap = cv2.VideoCapture(clip_path)
        tracks = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                faces = sample_face_frame(frame, self.face_model)
                for face in faces:
                    tracks.append({"frame": frame_idx, "box": face.box, "cx": face.cx})
            frame_idx += 1
        cap.release()
        return tracks

    def _render_single(self, clip_id, temp_clip, temp_ass, words,
                       asd_scores, src_w, src_h, fps, output_video) -> None:
        log.info("[%s] 📝 Membuat subtitle ASS...", clip_id)
        write_ass(words, temp_ass)
        del words

        centers, crop_stats = self._compute_single_crop(temp_clip, src_w, src_h, fps, asd_scores)
        log.info(
            "[%s] Crop stats — scene cuts: %d | hard jumps: %d | focus changes: %d | cut resets: %d",
            clip_id,
            crop_stats["scene_cuts"], crop_stats["hard_crop_jumps"],
            crop_stats["smooth_focus_changes"], crop_stats["source_cut_resets"],
        )
        log.info("[%s] 🎞️  Rendering final video...", clip_id)
        self.ffmpeg.composite(temp_clip, centers, src_w, src_h, temp_ass, output_video)
        del centers

    def _render_split(self, clip_id, temp_clip, temp_audio, temp_ass, words,
                      asd_scores, src_w, src_h, fps, output_video) -> None:
        log.info("[%s] Mode split screen — tracking dua speaker...", clip_id)
        centers_left, centers_right, centers_single, is_split, crop_stats = \
            self._compute_dual_crop(temp_clip, src_w, src_h, fps, asd_scores)

        bottom_margin = int(src_h * (1.0 - config.SPLIT_SCREEN_TOP_RATIO))
        log.info("[%s] 📝 Membuat subtitle ASS (margin bawah %dpx)...", clip_id, bottom_margin)
        write_ass(words, temp_ass, bottom_margin_px=bottom_margin)
        del words

        log.info("[%s] Crop stats — keyframes: %d", clip_id, crop_stats["target_keyframes"])
        log.info("[%s] 🎞️  Rendering split screen...", clip_id)
        self.ffmpeg.composite_split(
            temp_clip, centers_left, centers_right, centers_single,
            is_split, src_w, src_h, temp_ass, output_video,
        )
        del centers_left, centers_right

    def _compute_single_crop(self, clip_path, src_w, src_h, src_fps, asd_scores):
        """Inline single-speaker crop — delegates to reframe helpers."""
        from processing.reframe import compute_crop_centers_streaming
        return compute_crop_centers_streaming(
            clip_path, self.face_model, src_w, src_h, src_fps, asd_scores=asd_scores
        )

    def _compute_dual_crop(self, clip_path, src_w, src_h, src_fps, asd_scores):
        """Inline dual-speaker crop — delegates to reframe helpers."""
        from processing.reframe import compute_dual_crop_centers_streaming
        return compute_dual_crop_centers_streaming(
            clip_path, self.face_model, src_w, src_h, src_fps, asd_scores=asd_scores
        )
