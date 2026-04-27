import cv2
import numpy as np
from . import config


def _safe_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else 30.0


def _mouth_region(frame, box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    mx1 = int(x1 + w * 0.25)
    mx2 = int(x1 + w * 0.75)
    my1 = int(y1 + h * 0.60)
    my2 = int(y1 + h * 0.95)
    fh, fw = frame.shape[:2]
    mx1, mx2 = max(0, mx1), min(fw, mx2)
    my1, my2 = max(0, my1), min(fh, my2)
    if mx2 <= mx1 or my2 <= my1:
        return None
    crop = frame[my1:my2, mx1:mx2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def detect_faces(clip_path: str, face_model, sample_fps: float = None):
    if sample_fps is None:
        sample_fps = config.FACE_SAMPLE_FPS

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
                        "mouth_gray": _mouth_region(frame, (float(x1), float(y1), float(x2), float(y2))),
                    })
                faces.sort(key=lambda f: f["area"], reverse=True)
            results.append({"t": t, "frame": frame_idx, "faces": faces})
        frame_idx += 1

    cap.release()
    return results


def compute_speaking_scores(face_data, src_fps: float) -> list:
    smooth_window = max(1, int(
        src_fps * config.LIP_SMOOTH_SEC / max(1, src_fps / config.FACE_SAMPLE_FPS)
    ))

    tracks = []

    def _nearest_track(cx):
        if not tracks:
            return None, float("inf")
        dists = [abs(tr["cx"] - cx) for tr in tracks]
        idx = int(np.argmin(dists))
        return idx, dists[idx]

    scores_per_sample = []

    for sample in face_data:
        faces = sample["faces"]
        sample_scores = {}
        face_to_track = {}

        for fi, face in enumerate(faces):
            cx = face["cx"]
            tr_idx, dist = _nearest_track(cx)
            face_w = (face["box"][2] - face["box"][0]) if face.get("box") else 60
            if tr_idx is not None and dist < face_w * 0.75:
                face_to_track[fi] = tr_idx
            else:
                tracks.append({"cx": cx, "mouth_gray": None, "motion_history": []})
                face_to_track[fi] = len(tracks) - 1

        for fi, face in enumerate(faces):
            tr_idx = face_to_track[fi]
            tr = tracks[tr_idx]
            mouth = face.get("mouth_gray")
            motion = 0.0
            if mouth is not None and tr["mouth_gray"] is not None:
                prev = tr["mouth_gray"]
                if prev.shape == mouth.shape:
                    motion = float(np.mean(np.abs(mouth.astype(np.float32) - prev.astype(np.float32))))
                else:
                    resized = cv2.resize(prev, (mouth.shape[1], mouth.shape[0]))
                    motion = float(np.mean(np.abs(mouth.astype(np.float32) - resized.astype(np.float32))))

            tr["motion_history"].append(motion)
            if len(tr["motion_history"]) > smooth_window:
                tr["motion_history"].pop(0)
            smoothed = float(np.mean(tr["motion_history"]))

            tr["cx"] = face["cx"]
            tr["mouth_gray"] = mouth if mouth is not None else tr["mouth_gray"]
            sample_scores[fi] = smoothed

        scores_per_sample.append(sample_scores)

    return scores_per_sample


def _face_score(face, speaking_score: float = 0.0) -> float:
    size_score = face["area"] * (0.75 + face.get("conf", 1.0))
    if config.LIP_MOTION_WEIGHT <= 0.0 or speaking_score < config.LIP_MIN_MOTION:
        return size_score
    lip_norm = min(speaking_score / 20.0, 1.0)
    return (1.0 - config.LIP_MOTION_WEIGHT) * size_score + config.LIP_MOTION_WEIGHT * size_score * (1.0 + lip_norm * 3.0)


def pick_best_face(faces, sample_scores=None):
    if not faces:
        return None
    if sample_scores is None:
        return max(faces, key=_face_score)
    return max(
        enumerate(faces),
        key=lambda fi_f: _face_score(fi_f[1], sample_scores.get(fi_f[0], 0.0)),
    )[1]


def match_face_by_center(faces, current_cx, max_distance_px):
    if current_cx is None or not faces:
        return None
    nearest = min(faces, key=lambda f: abs(f["cx"] - current_cx))
    if abs(nearest["cx"] - current_cx) <= max_distance_px:
        return nearest
    return None
