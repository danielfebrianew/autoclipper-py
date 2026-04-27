import os
from dotenv import load_dotenv

load_dotenv()

json_file     = os.environ.get("AUTOCLIPPER_JSON", "")
video_file    = os.environ.get("AUTOCLIPPER_VIDEO", "")
out_dir       = os.environ.get("AUTOCLIPPER_OUTDIR", ".")
channel_name  = os.environ.get("AUTOCLIPPER_CHANNEL", "")
source_credit = os.environ.get("AUTOCLIPPER_SOURCE_CREDIT", "")

# Root of the project (one level up from processing/)
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_WORDS_PER_SCREEN = 2

FACE_SAMPLE_FPS = float(os.environ.get("AUTOCLIPPER_FACE_SAMPLE_FPS", "4"))

SCENE_CUT_SCORE_THRESHOLD = float(os.environ.get("AUTOCLIPPER_SCENE_CUT_SCORE", "0.22"))
SCENE_CUT_HIST_THRESHOLD  = float(os.environ.get("AUTOCLIPPER_SCENE_CUT_HIST",  "0.14"))
SCENE_CUT_PIXEL_THRESHOLD = float(os.environ.get("AUTOCLIPPER_SCENE_CUT_PIXEL", "0.08"))
SCENE_CUT_MIN_GAP_SEC     = float(os.environ.get("AUTOCLIPPER_SCENE_CUT_MIN_GAP", "0.30"))

FOCUS_MIN_LOCK_SEC          = float(os.environ.get("AUTOCLIPPER_FOCUS_MIN_LOCK",       "1.50"))
FOCUS_SWITCH_CONFIRM_SEC    = float(os.environ.get("AUTOCLIPPER_FOCUS_CONFIRM",        "0.85"))
FOCUS_SWITCH_AREA_RATIO     = float(os.environ.get("AUTOCLIPPER_FOCUS_AREA_RATIO",     "1.35"))
FOCUS_LOST_GRACE_SEC        = float(os.environ.get("AUTOCLIPPER_FOCUS_LOST_GRACE",     "0.80"))
FOCUS_MATCH_DISTANCE_RATIO  = float(os.environ.get("AUTOCLIPPER_FOCUS_MATCH_DISTANCE", "0.35"))

CROP_DEADZONE_RATIO        = float(os.environ.get("AUTOCLIPPER_CROP_DEADZONE",         "0.07"))
CROP_MIN_DEADZONE_PX       = float(os.environ.get("AUTOCLIPPER_CROP_MIN_DEADZONE_PX",  "36"))
CROP_SMOOTHING_TAU_SEC     = float(os.environ.get("AUTOCLIPPER_CROP_SMOOTHING_TAU",    "0.45"))
CROP_MAX_SPEED_PX_PER_SEC  = float(os.environ.get("AUTOCLIPPER_CROP_MAX_SPEED",        "480"))

LIP_CUT_COOLDOWN_1_SEC  = float(os.environ.get("AUTOCLIPPER_LIP_COOLDOWN_1",    "1.0"))
LIP_CUT_COOLDOWN_2_SEC  = float(os.environ.get("AUTOCLIPPER_LIP_COOLDOWN_2",    "2.0"))
LIP_CUT_COOLDOWN_3_SEC  = float(os.environ.get("AUTOCLIPPER_LIP_COOLDOWN_3",    "1.0"))
LIP_CUT_BURST_WINDOW_SEC = float(os.environ.get("AUTOCLIPPER_LIP_BURST_WINDOW", "4.0"))

LIP_MOTION_WEIGHT = float(os.environ.get("AUTOCLIPPER_LIP_MOTION_WEIGHT", "0.6"))
LIP_SMOOTH_SEC    = float(os.environ.get("AUTOCLIPPER_LIP_SMOOTH_SEC",    "0.5"))
LIP_MIN_MOTION    = float(os.environ.get("AUTOCLIPPER_LIP_MIN_MOTION",    "3.0"))
