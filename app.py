"""
Wire Inspector — Dual-Camera YOLOv8 Region Detection Server
=============================================================
Loads two YOLOv8 models (one per camera) and runs region detection
on captured frames.

Stage-2 wire-presence classifier is not yet trained — detected regions
are reported as "detected" only.

Usage:
    python app.py --cam1-camera 0 --cam2-camera 1
    python app.py --cam1-model checkpoints/cam1.pt --cam2-model checkpoints/cam2.pt

Then open http://YOUR_IP:5000 in your browser.
"""

import sys
import atexit
import argparse
import socket
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `import state`, `import config` work

from flask import Flask, request

import state
from config import (
    DEFAULT_CAM1_MODEL, DEFAULT_CAM2_MODEL,
    DEFAULT_CAM1_ANOMALY, DEFAULT_CAM2_ANOMALY,
)
from models.detector import YoloDualDetector
from models.camera   import CameraManager
from models.anomaly  import PatchCoreDetector

from api.camera    import bp as camera_bp
from api.analyze   import bp as analyze_bp
from api.calibrate import bp as calibrate_bp
from api.views     import bp as views_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("yolo_app")

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


@app.before_request
def _log_request():
    if request.path == "/camera-status":
        return
    log.debug(">> %s %s", request.method, request.path)


app.register_blueprint(camera_bp)
app.register_blueprint(analyze_bp)
app.register_blueprint(calibrate_bp)
app.register_blueprint(views_bp)


def main():
    parser = argparse.ArgumentParser(description="Wire Inspector — YOLOv8 Dual Camera")
    parser.add_argument("--cam1-model", type=str, default=str(DEFAULT_CAM1_MODEL),
                        help="Path to cam1 YOLOv8 checkpoint (.pt)")
    parser.add_argument("--cam2-model", type=str, default=str(DEFAULT_CAM2_MODEL),
                        help="Path to cam2 YOLOv8 checkpoint (.pt)")
    parser.add_argument("--cam1-anomaly", type=str, default=str(DEFAULT_CAM1_ANOMALY),
                        help="Path to cam1 PatchCore checkpoint (.pkl)")
    parser.add_argument("--cam2-anomaly", type=str, default=str(DEFAULT_CAM2_ANOMALY),
                        help="Path to cam2 PatchCore checkpoint (.pkl)")
    parser.add_argument("--cam1-camera", type=int, default=0, help="USB index for cam1")
    parser.add_argument("--cam2-camera", type=int, default=1, help="USB index for cam2")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # ── Detector ──────────────────────────────────────────────────────────
    cam1_path = Path(args.cam1_model)
    cam2_path = Path(args.cam2_model)
    if not cam1_path.exists():
        print(f"ERROR: cam1 model not found at {cam1_path}")
        print(f"  Drop it there or pass --cam1-model")
        return
    if not cam2_path.exists():
        print(f"ERROR: cam2 model not found at {cam2_path}")
        print(f"  Drop it there or pass --cam2-model")
        return

    print("Loading YOLOv8 models...")
    state.detector = YoloDualDetector(str(cam1_path), str(cam2_path))
    print(f"  cam1: {cam1_path.name}  ({len(state.detector.cam1_classes)} classes)")
    print(f"  cam2: {cam2_path.name}  ({len(state.detector.cam2_classes)} classes)")
    print(f"  Device: {state.detector.device}")

    # ── PatchCore anomaly models (optional) ───────────────────────────────
    cam1_anom = Path(args.cam1_anomaly)
    cam2_anom = Path(args.cam2_anomaly)
    cam1_anom_path = str(cam1_anom) if cam1_anom.exists() else None
    cam2_anom_path = str(cam2_anom) if cam2_anom.exists() else None

    if cam1_anom_path or cam2_anom_path:
        print("Loading PatchCore anomaly models...")
        state.anomaly_detector = PatchCoreDetector(cam1_anom_path, cam2_anom_path)
        if state.anomaly_detector.has_cam1:
            print(f"  cam1: {cam1_anom.name}  ({state.anomaly_detector.region_count('cam1')} region banks)")
        else:
            print(f"  cam1: not loaded")
        if state.anomaly_detector.has_cam2:
            print(f"  cam2: {cam2_anom.name}  ({state.anomaly_detector.region_count('cam2')} region banks)")
        else:
            print(f"  cam2: not loaded")
    else:
        print("PatchCore anomaly: not configured (no .pkl files found)")

    # ── Cameras ───────────────────────────────────────────────────────────
    print("Initializing cameras...")
    state.camera_manager = CameraManager(args.cam1_camera, args.cam2_camera)
    cam_status = state.camera_manager.open()
    print(f"  cam1 (index {args.cam1_camera}): {'OK' if cam_status['cam1'] else 'FAILED'}")
    print(f"  cam2 (index {args.cam2_camera}): {'OK' if cam_status['cam2'] else 'FAILED'}")

    def cleanup():
        if state.camera_manager:
            state.camera_manager.release()
            print("Cameras released.")
    atexit.register(cleanup)

    # ── Server ────────────────────────────────────────────────────────────
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    print("=" * 55)
    print("Wire Inspector — Dual Camera YOLOv8")
    print("=" * 55)
    print(f"Open in browser: http://{local_ip}:{args.port}")
    print("=" * 55)

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
