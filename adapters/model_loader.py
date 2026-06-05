"""Model loader adapter — centralizes all ML model loading."""

import os

from core.exceptions import ModelLoadError
from processing.logger import get_logger

log = get_logger("adapters.model_loader")


class ModelLoader:
    def __init__(self, app_dir: str):
        self.app_dir = app_dir

    def load_whisper(self):
        from faster_whisper import WhisperModel
        log.info("Memuat model Faster-Whisper (medium) di CPU...")
        return WhisperModel("medium", device="cpu", compute_type="int8")

    def load_face_detector(self):
        from ultralytics import YOLO
        weights = os.path.join(self.app_dir, "yolov8n-face-lindevs.pt")
        log.info("Memuat model YOLOv8 face detection...")
        return YOLO(weights)

    def load_asd(self) -> tuple:
        """Return (model, device) or (None, None) on failure."""
        from processing.asd import load_asd_model
        log.info("Memuat model Light-ASD (active speaker detection)...")
        try:
            return load_asd_model()
        except Exception:
            log.exception("Gagal memuat Light-ASD, ASD dinonaktifkan")
            return None, None
