import subprocess
import cv2
import numpy as np
from . import config
from .logger import get_logger

log = get_logger("processing.ffmpeg")


def cut_clip(start, duration, src_video: str, dest: str) -> None:
    result = subprocess.run([
        "ffmpeg", "-ss", str(start), "-i", src_video,
        "-t", str(duration), "-c", "copy", "-y", dest,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[-1000:]
        log.error("ffmpeg cut_clip gagal (rc=%d):\n%s", result.returncode, err)
        raise subprocess.CalledProcessError(result.returncode, "ffmpeg", stderr=err)


def extract_audio(src_video: str, dest_wav: str) -> None:
    subprocess.run([
        "ffmpeg", "-i", src_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", dest_wav,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def composite(temp_clip: str, centers: np.ndarray, src_w: int, src_h: int,
              ass_path: str, output_video: str) -> None:
    crop_w = int(src_h * 9 / 16)

    cap = cv2.VideoCapture(temp_clip)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    safe_ass = ass_path.replace("\\", "/").replace(":", "\\:")
    filters = [f"ass={safe_ass}"]

    if config.source_credit.strip():
        txt = _escape_drawtext(config.source_credit.strip())
        filters.append(
            f"drawtext=text='{txt}':font=Impact:fontsize=28"
            f":fontcolor=white@0.55:x=(w-text_w)/2:y=40"
            f":shadowcolor=black@0.6:shadowx=2:shadowy=2"
        )

    if config.channel_name.strip():
        txt = _escape_drawtext(config.channel_name.strip())
        filters.insert(1,
            f"drawtext=text='{txt}':font=Impact:fontsize=28"
            f":fontcolor=white@0.20:x=(w-text_w)/2:y=(h-text_h)/2"
            f":shadowcolor=black@0.20:shadowx=2:shadowy=2"
        )

    vf = ",".join(filters)

    proc = subprocess.Popen([
        "ffmpeg",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{crop_w}x{src_h}",
        "-pix_fmt", "bgr24",
        "-r", f"{src_fps}",
        "-i", "-",
        "-i", temp_clip,
        "-map", "0:v", "-map", "1:a",
        "-vf", vf,
        "-c:v", "h264_videotoolbox",
        "-pix_fmt", "yuv420p",
        "-b:v", "5000k",
        "-c:a", "aac",
        "-shortest",
        "-y", output_video,
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
        err_tail = (stderr or b"").decode("utf-8", errors="replace")[-1500:]
        log.error("ffmpeg composite gagal (rc=%d):\n%s", proc.returncode, err_tail)
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, output=None, stderr=err_tail,
        )


def composite_split(temp_clip: str, centers_left: np.ndarray, centers_right: np.ndarray,
                    src_w: int, src_h: int, ass_path: str, output_video: str) -> None:
    """
    Split-screen composite: top 60% has two side-by-side speaker panels,
    bottom 40% is black (for external image overlay in post).
    """
    panel_w = int(src_h * 9 / 16) // 2
    out_w   = panel_w * 2          # guaranteed even, matches hstack output
    top_h   = int(src_h * config.SPLIT_SCREEN_TOP_RATIO)

    cap     = cv2.VideoCapture(temp_clip)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    safe_ass = ass_path.replace("\\", "/").replace(":", "\\:")
    filters  = [f"ass={safe_ass}"]

    if config.source_credit.strip():
        txt = _escape_drawtext(config.source_credit.strip())
        filters.append(
            f"drawtext=text='{txt}':font=Impact:fontsize=24"
            f":fontcolor=white@0.55:x=(w-text_w)/2:y=30"
            f":shadowcolor=black@0.6:shadowx=2:shadowy=2"
        )

    if config.channel_name.strip():
        txt = _escape_drawtext(config.channel_name.strip())
        filters.insert(1,
            f"drawtext=text='{txt}':font=Impact:fontsize=24"
            f":fontcolor=white@0.20:x=(w-text_w)/2:y=(h-text_h)/2"
            f":shadowcolor=black@0.20:shadowx=2:shadowy=2"
        )

    vf = ",".join(filters)

    proc = subprocess.Popen([
        "ffmpeg",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{src_h}",
        "-pix_fmt", "bgr24",
        "-r", f"{src_fps}",
        "-i", "-",
        "-i", temp_clip,
        "-map", "0:v", "-map", "1:a",
        "-vf", vf,
        "-c:v", "h264_videotoolbox",
        "-pix_fmt", "yuv420p",
        "-b:v", "5000k",
        "-c:a", "aac",
        "-shortest",
        "-y", output_video,
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    assert proc.stdin is not None

    bottom = np.zeros((src_h - top_h, out_w, 3), dtype=np.uint8)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            i = min(frame_idx, len(centers_left) - 1)
            cx_l = centers_left[i]
            cx_r = centers_right[i]

            x1_l = int(np.clip(cx_l - panel_w / 2, 0, src_w - panel_w))
            x1_r = int(np.clip(cx_r - panel_w / 2, 0, src_w - panel_w))

            left_crop  = frame[:top_h, x1_l:x1_l + panel_w]
            right_crop = frame[:top_h, x1_r:x1_r + panel_w]

            top      = np.hstack([left_crop, right_crop])
            out_frame = np.vstack([top, bottom])
            proc.stdin.write(out_frame.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass

    _, stderr = proc.communicate()
    if proc.returncode != 0:
        err_tail = (stderr or b"").decode("utf-8", errors="replace")[-1500:]
        log.error("ffmpeg composite_split gagal (rc=%d):\n%s", proc.returncode, err_tail)
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, output=None, stderr=err_tail,
        )
