import cv2
from . import config


def _safe_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else 30.0


def sample_face_frame(frame, face_model) -> list:
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
                "cx":   float((x1 + x2) / 2),
                "cy":   float((y1 + y2) / 2),
                "area": float(area),
                "conf": float(conf),
                "box":  (float(x1), float(y1), float(x2), float(y2)),
            })
        faces.sort(key=lambda f: f["area"], reverse=True)
        faces = faces[:config.MAX_FACES_PER_SAMPLE]
    return faces


def _face_score(face, asd_score: float = 0.0) -> float:
    base = face["area"] * (0.75 + face.get("conf", 1.0))
    # ASD speaking boost: up to 3× base score when model is confident (score=1.0)
    return base * (1.0 + 2.0 * asd_score)


def pick_best_face(faces, asd_scores: dict | None = None):
    """
    asd_scores: optional dict mapping face cx (rounded int) → speaking probability.
    When provided, faces with high speaking score are strongly preferred.
    """
    if not faces:
        return None
    if not asd_scores:
        return max(faces, key=_face_score)

    def scored(face):
        key = round(face["cx"])
        # find nearest cx key within 50px
        best_asd = 0.0
        for cx_key, prob in asd_scores.items():
            if abs(cx_key - key) < 50:
                best_asd = max(best_asd, prob)
        return _face_score(face, best_asd)

    return max(faces, key=scored)


def match_face_by_center(faces, current_cx, max_distance_px):
    if current_cx is None or not faces:
        return None
    nearest = min(faces, key=lambda f: abs(f["cx"] - current_cx))
    if abs(nearest["cx"] - current_cx) <= max_distance_px:
        return nearest
    return None
