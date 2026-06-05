from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel


# ── Face Detection ────────────────────────────────────────────────────────────

class Face(BaseModel):
    cx: float
    cy: float
    area: float
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def score(self) -> float:
        return self.area * (0.75 + self.conf)

    def asd_score(self, asd_scores: dict | None, radius_px: float = 50.0) -> float:
        if not asd_scores:
            return 0.0
        key = round(self.cx)
        best = 0.0
        for cx_key, prob in asd_scores.items():
            if abs(cx_key - key) < radius_px:
                best = max(best, prob)
        return best

    def weighted_score(self, asd_scores: dict | None = None) -> float:
        return self.score * (1.0 + 2.0 * self.asd_score(asd_scores))


# ── Reframe ───────────────────────────────────────────────────────────────────

class Keyframe(BaseModel):
    frame: int
    cx: float
    hard: bool = False


# ── Clip ──────────────────────────────────────────────────────────────────────

class Clip(BaseModel):
    clip_id: int
    start_time: str
    end_time: str
    start_seconds: int
    end_seconds: int
    duration_seconds: int
    suggested_caption: str
    viral_score: int
    speaker: str = ""
    speakers_visible: list[str] = []
    interaction_type: str = ""
    hook: str = ""
    summary: str = ""
    category: str = ""
    energy_level: str = ""
    transcript_excerpt: str = ""
    end_cue: str = ""
    timestamp_adjustments: list[str] = []


class Speaker(BaseModel):
    name: str
    role: str = ""
    position: str = ""
    description: str = ""


class ClipsJson(BaseModel):
    clips: list[Clip]
    speakers: list[Speaker] = []
    video_title: str = ""
    video_duration: str = ""
    video_duration_seconds: int = 0
    source_url: str = ""
    heatmap_available: bool = False


# ── Render Job ────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"


@dataclass
class RenderJob:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    logs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
