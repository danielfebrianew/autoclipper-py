import cv2
from . import config
from core.models import Face


def _safe_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else 30.0


def sample_face_frame(frame, face_model) -> list[Face]:
    detections = face_model(frame, verbose=False)[0]
    faces: list[Face] = []
    if len(detections.boxes) > 0:
        boxes = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            area = max(0.0, (x2 - x1) * (y2 - y1))
            if area <= 0:
                continue
            faces.append(Face(
                cx=float((x1 + x2) / 2),
                cy=float((y1 + y2) / 2),
                area=float(area),
                conf=float(conf),
                box=(float(x1), float(y1), float(x2), float(y2)),
            ))
        faces.sort(key=lambda f: f.area, reverse=True)
        faces = faces[:config.MAX_FACES_PER_SAMPLE]
    return faces


def pick_best_face(faces: list[Face], asd_scores: dict | None = None) -> Face | None:
    if not faces:
        return None
    return max(faces, key=lambda f: f.weighted_score(asd_scores))


def match_face_by_center(faces: list[Face], current_cx: float, max_distance_px: float) -> Face | None:
    if current_cx is None or not faces:
        return None
    nearest = min(faces, key=lambda f: abs(f.cx - current_cx))
    if abs(nearest.cx - current_cx) <= max_distance_px:
        return nearest
    return None
