"""App configuration."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# Where the trained YOLO checkpoints live.
# Drop your trained models here as cam1.pt and cam2.pt (or pass --cam1-model / --cam2-model).
CHECKPOINTS_DIR = APP_DIR / "checkpoints"
DEFAULT_CAM1_MODEL = CHECKPOINTS_DIR / "cam1.pt"
DEFAULT_CAM2_MODEL = CHECKPOINTS_DIR / "cam2.pt"

# PatchCore anomaly model checkpoints (.pkl from train_patchcore.py)
DEFAULT_CAM1_ANOMALY = CHECKPOINTS_DIR / "cam1_patchcore.pkl"
DEFAULT_CAM2_ANOMALY = CHECKPOINTS_DIR / "cam2_patchcore.pkl"

# Multiplies the per-region threshold from training. >1.0 = less sensitive,
# <1.0 = more sensitive. The .pkl threshold is mean + 5σ; multiplying by 1.5
# behaves like mean + 7.5σ, etc.
ANOMALY_THRESHOLD_MULTIPLIER = 1.5

# Inference
INFERENCE_IMG_SIZE = 1920     # MUST match training imgsz (check runs/cam1/args.yaml)
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Camera capture
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080

# Class list — must match prepare_yolo.py
BOX_LABELS = [
    "Row 1 Top", "Row 1 Bottom", "Rows 2-5 Top",
    "Row 2 Bottom", "Row 3 Bottom", "Row 4 Bottom", "Row 5 Bottom",
    "Row 6 Top", "Row 7 Top", "Row 8 Top", "Row 9 Top", "Rows 6-9 Bottom",
    "Row 10 Top", "Row 10 Bottom",
    "Row 11 Top", "Row 11 Bottom",
    "Row 12 Top",
    "Row 13 Top", "Row 13 Bottom",
    "Row 14 Top", "Row 14 Bottom",
    "Row 15 Top", "Row 15 Bottom",
]
