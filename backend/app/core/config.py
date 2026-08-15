"""
OGB — OrbitalGuard
Core configuration: risk thresholds, model paths, and app settings.
All risk thresholds are defined here — never hard-coded throughout the app.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]  # repo root
MODEL_WEIGHTS_PATH = ROOT_DIR / "ml" / "weights" / "ogb_yolov8n.pt"

# ---------------------------------------------------------------------------
# YOLO / CV
# ---------------------------------------------------------------------------
YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_MAX_DETECTIONS = 100

# 11 class names — exact order must match data.yaml (index = class_id)
CLASS_NAMES = [
    "cheops",
    "debris",
    "double_start",
    "earth_observation_sat_1",
    "lisa_pathfinder",
    "proba_2",
    "proba_3_csc",
    "proba_3_ocs",
    "smart_1",
    "soho",
    "xmm_newton",
]

# ---------------------------------------------------------------------------
# Risk engine thresholds  (score range → category)
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "LOW":      (0.00, 0.24),
    "MEDIUM":   (0.25, 0.49),
    "HIGH":     (0.50, 0.74),
    "CRITICAL": (0.75, 1.00),
}

RISK_CATEGORY_EMOJI = {
    "LOW":      "🟢",
    "MEDIUM":   "🟡",
    "HIGH":     "🟠",
    "CRITICAL": "🔴",
}

# Hard-body radius used in risk formula (km).
# Default: representative value for a 10 m-class object.
HARD_BODY_RADIUS_KM: float = 0.010

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_V1_PREFIX = "/api/v1"
APP_TITLE = "OGB — OrbitalGuard API"
APP_VERSION = "0.1.0"
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# ---------------------------------------------------------------------------
# AI Copilot
# ---------------------------------------------------------------------------
COPILOT_PROVIDER = "gemini"          # swap here to change provider globally
COPILOT_MODEL = "gemini-3.5-flash"
COPILOT_MAX_TOKENS = 1024
