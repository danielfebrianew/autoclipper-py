"""
Light-ASD wrapper — Active Speaker Detection.

Model: Light-ASD (CVPR 2023) by Junhua Liao et al.
Weights: downloaded on first use to APP_DIR/weights/light_asd.model

Input per face per window:
  - visual: grayscale 112x112 face crops, shape (1, T, 112, 112)
  - audio:  MFCC 13-coeff, shape (1, T*4, 13)

Output: per-frame speaking score (float, higher = more likely speaking)
"""

import os
import urllib.request
from .logger import get_logger

log = get_logger("processing.asd")
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from python_speech_features import mfcc

from . import config

# ---------------------------------------------------------------------------
# Pretrained weights
# ---------------------------------------------------------------------------
_WEIGHTS_URL = (
    "https://github.com/Junhua-Liao/Light-ASD/raw/main/weight/pretrain_AVA_CVPR.model"
)
_WEIGHTS_PATH = os.path.join(config.APP_DIR, "weights", "light_asd.model")


def _ensure_weights():
    os.makedirs(os.path.dirname(_WEIGHTS_PATH), exist_ok=True)
    if not os.path.exists(_WEIGHTS_PATH):
        log.info("Downloading Light-ASD weights (~20 MB)...")
        urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS_PATH)
        log.info("Download selesai.")


# ---------------------------------------------------------------------------
# Model architecture (replicated from Light-ASD repo)
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class _VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.layer1 = nn.Sequential(_BasicBlock(64, 64), _BasicBlock(64, 64))
        self.layer2 = nn.Sequential(_BasicBlock(64, 128, stride=2), _BasicBlock(128, 128))
        self.layer3 = nn.Sequential(_BasicBlock(128, 256, stride=2), _BasicBlock(256, 256))
        self.layer4 = nn.Sequential(_BasicBlock(256, 512, stride=2), _BasicBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x: (B, T, 112, 112) → add channel dim → (B, 1, T, 112, 112)
        x = x.unsqueeze(1)
        x = self.frontend(x)
        B, C, T, H, W = x.shape
        # fold time into batch for 2-D ResNet blocks
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)          # (B*T, 512)
        x = x.reshape(B, T, 512)
        return x


class _AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 128)
        self.fc2 = nn.Linear(128, 512)

    def forward(self, x):
        # x: (B, T*4, 13) — pool every 4 frames to match visual T
        B, A, D = x.shape
        T = A // 4
        x = x[:, :T * 4, :].reshape(B, T, 4, D).mean(dim=2)  # (B, T, 13)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)                         # (B, T, 512)
        return x


class _LightASD(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = _VisualEncoder()
        self.audio_encoder  = _AudioEncoder()
        self.classifier     = nn.Linear(1024, 2)

    def forward(self, visual, audio):
        v = self.visual_encoder(visual)   # (B, T, 512)
        a = self.audio_encoder(audio)     # (B, T, 512)
        x = torch.cat([v, a], dim=-1)     # (B, T, 1024)
        return self.classifier(x)         # (B, T, 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_asd_model():
    """Load Light-ASD model onto the best available device. Returns (model, device)."""
    _ensure_weights()

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    log.info("Menggunakan device: %s", device)

    model = _LightASD()
    state = torch.load(_WEIGHTS_PATH, map_location="cpu")
    # strip "module." prefix if saved with DataParallel
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Load only matching keys — architecture may differ slightly from AVA checkpoint.
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(matched)
    model.load_state_dict(own)
    model.to(device).eval()
    log.info("Model siap (%d/%d weights dimuat).", len(matched), len(own))
    if len(matched) < len(own) * 0.5:
        log.warning("Kurang dari 50%% weights cocok — arsitektur checkpoint mungkin berbeda.")
    return model, device


def compute_asd_scores(
    clip_path: str,
    audio_wav: str,
    face_tracks: list[dict],
    model: nn.Module,
    device: torch.device,
    src_fps: float,
    window_sec: float = 0.5,
) -> dict[int, float]:
    """
    Compute per-face-track speaking score across the clip.

    face_tracks: list of {"frame": int, "face_idx": int, "box": (x1,y1,x2,y2)}
                 — only faces already picked by the face detector.

    Returns: dict mapping frame_num → speaking_score (0.0–1.0)
             for the most-likely-speaking face at each sampled frame.
    """
    import scipy.io.wavfile as wavfile

    if not face_tracks:
        return {}

    sample_rate, audio_data = wavfile.read(audio_wav)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    audio_data = audio_data.astype(np.float32) / 32768.0

    cap = cv2.VideoCapture(clip_path)
    window_frames = max(1, int(src_fps * window_sec))

    # Group tracks by frame
    by_frame: dict[int, list[dict]] = {}
    for t in face_tracks:
        by_frame.setdefault(t["frame"], []).append(t)

    sorted_frames = sorted(by_frame.keys())
    frame_scores: dict[int, float] = {}

    for center_frame in sorted_frames:
        tracks = by_frame[center_frame]
        half = window_frames // 2
        f_start = max(0, center_frame - half)
        f_end   = center_frame + half

        # --- audio window ---
        t_start = f_start / src_fps
        t_end   = f_end   / src_fps
        a_start = int(t_start * sample_rate)
        a_end   = int(t_end   * sample_rate)
        audio_chunk = audio_data[a_start:a_end]
        if len(audio_chunk) < sample_rate // 10:
            continue

        mfcc_feat = mfcc(audio_chunk, sample_rate, numcep=13, nfft=512)  # (A, 13)
        if mfcc_feat.shape[0] < 4:
            continue

        best_score = 0.0
        for track in tracks:
            x1, y1, x2, y2 = (int(v) for v in track["box"])

            # --- visual window: read frames around center ---
            crops = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
            for _ in range(f_end - f_start):
                ret, fr = cap.read()
                if not ret:
                    break
                face_crop = fr[max(0,y1):y2, max(0,x1):x2]
                if face_crop.size == 0:
                    face_crop = np.zeros((112, 112), dtype=np.uint8)
                else:
                    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    face_crop = cv2.resize(face_crop, (112, 112))
                crops.append(face_crop)

            if not crops:
                continue

            T = len(crops)
            visual = np.stack(crops).astype(np.float32) / 255.0  # (T, 112, 112)
            A = T * 4
            mfcc_trimmed = mfcc_feat[:A] if mfcc_feat.shape[0] >= A else np.pad(
                mfcc_feat, ((0, A - mfcc_feat.shape[0]), (0, 0))
            )

            v_tensor = torch.from_numpy(visual.astype(np.float32)).unsqueeze(0).to(device)
            a_tensor = torch.from_numpy(mfcc_trimmed.astype(np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(v_tensor, a_tensor)          # (1, T, 2)
                probs  = torch.softmax(logits, dim=-1)      # (1, T, 2)
                score  = float(probs[0, :, 1].mean().cpu()) # mean P(speaking)

            best_score = max(best_score, score)

        frame_scores[center_frame] = best_score

    cap.release()
    return frame_scores
