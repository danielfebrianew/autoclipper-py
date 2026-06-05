"""FFmpeg adapter — thin wrapper around processing/ffmpeg_utils.py."""

import numpy as np

from processing.ffmpeg_utils import (
    cut_clip,
    extract_audio,
    composite,
    composite_split,
)
from core.exceptions import FFmpegError


class FFmpegAdapter:
    def cut(self, start: str, duration: str, src: str, dest: str) -> None:
        import subprocess
        try:
            cut_clip(start, duration, src, dest)
        except subprocess.CalledProcessError as e:
            raise FFmpegError("cut gagal", e.returncode, str(e.stderr or ""))

    def extract_audio(self, src: str, dest: str) -> None:
        extract_audio(src, dest)

    def composite(self, temp_clip: str, centers: np.ndarray,
                  src_w: int, src_h: int, ass_path: str, output: str) -> None:
        composite(temp_clip, centers, src_w, src_h, ass_path, output)

    def composite_split(self, temp_clip: str, centers_left: np.ndarray,
                        centers_right: np.ndarray, centers_single: np.ndarray,
                        is_split: np.ndarray, src_w: int, src_h: int,
                        ass_path: str, output: str) -> None:
        composite_split(temp_clip, centers_left, centers_right, centers_single,
                        is_split, src_w, src_h, ass_path, output)
