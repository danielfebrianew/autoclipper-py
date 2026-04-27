import subprocess
import cv2
import numpy as np
from . import config


def cut_clip(start, duration, src_video: str, dest: str) -> None:
    result = subprocess.run([
        "ffmpeg", "-ss", str(start), "-i", src_video,
        "-t", str(duration), "-c", "copy", "-y", dest,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[-1000:]
        print(f"[ffmpeg stderr]\n{err}", flush=True)
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
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, output=None, stderr=err_tail,
        )
