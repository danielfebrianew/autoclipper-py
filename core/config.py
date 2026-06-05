import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


@dataclass
class PathConfig:
    json_file:     str = field(default_factory=lambda: os.environ.get("AUTOCLIPPER_JSON", ""))
    video_file:    str = field(default_factory=lambda: os.environ.get("AUTOCLIPPER_VIDEO", ""))
    out_dir:       str = field(default_factory=lambda: os.environ.get("AUTOCLIPPER_OUTDIR", "."))
    channel_name:  str = field(default_factory=lambda: os.environ.get("AUTOCLIPPER_CHANNEL", ""))
    source_credit: str = field(default_factory=lambda: os.environ.get("AUTOCLIPPER_SOURCE_CREDIT", ""))
    render_mode:   str = field(default_factory=lambda: os.environ.get("AUTOCLIPPER_MODE", "single"))
    app_dir:       str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class FaceConfig:
    sample_fps:     float = field(default_factory=lambda: _env_float("AUTOCLIPPER_FACE_SAMPLE_FPS", 4.0))
    max_per_sample: int   = field(default_factory=lambda: _env_int("AUTOCLIPPER_MAX_FACES", 3))


@dataclass
class SceneCutConfig:
    score_threshold: float = field(default_factory=lambda: _env_float("AUTOCLIPPER_SCENE_CUT_SCORE", 0.22))
    hist_threshold:  float = field(default_factory=lambda: _env_float("AUTOCLIPPER_SCENE_CUT_HIST",  0.14))
    pixel_threshold: float = field(default_factory=lambda: _env_float("AUTOCLIPPER_SCENE_CUT_PIXEL", 0.08))
    min_gap_sec:     float = field(default_factory=lambda: _env_float("AUTOCLIPPER_SCENE_CUT_MIN_GAP", 0.30))


@dataclass
class FocusConfig:
    min_lock_sec:         float = field(default_factory=lambda: _env_float("AUTOCLIPPER_FOCUS_MIN_LOCK",       1.50))
    switch_confirm_sec:   float = field(default_factory=lambda: _env_float("AUTOCLIPPER_FOCUS_CONFIRM",        0.85))
    switch_area_ratio:    float = field(default_factory=lambda: _env_float("AUTOCLIPPER_FOCUS_AREA_RATIO",     1.35))
    lost_grace_sec:       float = field(default_factory=lambda: _env_float("AUTOCLIPPER_FOCUS_LOST_GRACE",     0.80))
    match_distance_ratio: float = field(default_factory=lambda: _env_float("AUTOCLIPPER_FOCUS_MATCH_DISTANCE", 0.35))


@dataclass
class CropConfig:
    deadzone_ratio:       float = field(default_factory=lambda: _env_float("AUTOCLIPPER_CROP_DEADZONE",          0.07))
    min_deadzone_px:      float = field(default_factory=lambda: _env_float("AUTOCLIPPER_CROP_MIN_DEADZONE_PX",   36.0))
    smoothing_tau_sec:    float = field(default_factory=lambda: _env_float("AUTOCLIPPER_CROP_SMOOTHING_TAU",     0.45))
    max_speed_px_per_sec: float = field(default_factory=lambda: _env_float("AUTOCLIPPER_CROP_MAX_SPEED",        480.0))
    split_top_ratio:      float = field(default_factory=lambda: _env_float("AUTOCLIPPER_SPLIT_TOP_RATIO",        0.60))
    max_words_per_screen: int   = 2


@dataclass
class LLMConfig:
    api_key:           str   = field(default_factory=lambda: os.environ.get("KIE_AI_API_KEY", ""))
    endpoint:          str   = "https://api.kie.ai/gemini-3-flash/v1/chat/completions"
    max_retries:       int   = 3
    retry_delay_sec:   float = 5.0
    max_transcript_words: int = 15000


@dataclass
class ClipGenConfig:
    default_max_clips:   int   = 10
    default_min_duration: int  = 20
    default_max_duration: int  = 90
    default_buffer:      int   = 5
    default_language:    str   = "id"
    heatmap_peak_threshold: float = 0.5
    heatmap_high_threshold: float = 0.7


@dataclass
class AppConfig:
    paths:    PathConfig    = field(default_factory=PathConfig)
    face:     FaceConfig    = field(default_factory=FaceConfig)
    scene_cut: SceneCutConfig = field(default_factory=SceneCutConfig)
    focus:    FocusConfig   = field(default_factory=FocusConfig)
    crop:     CropConfig    = field(default_factory=CropConfig)
    llm:      LLMConfig     = field(default_factory=LLMConfig)
    clip_gen: ClipGenConfig = field(default_factory=ClipGenConfig)


_default_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _default_config
    if _default_config is None:
        _default_config = AppConfig()
    return _default_config
