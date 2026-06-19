"""Calibration endpoints — analyze a folder of normal images and suggest
threshold multipliers so all those samples register as PASS."""
import logging
import time

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

import state

bp = Blueprint("calibrate", __name__)
log = logging.getLogger("yolo_app.calibrate")

# How much margin to add above the worst observed ratio (1.10 = 10% safety)
SAFETY_MARGIN = 1.10


def _decode(file_storage):
    if file_storage is None:
        return None
    buf = np.frombuffer(file_storage.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _gather_ratios(cam: str, files, detect_fn) -> dict:
    """Run YOLO + raw PatchCore on every file, collect ratios per region."""
    region_ratios: dict[str, list[float]] = {}
    images_used = 0
    detections_total = 0

    for fs in files:
        img = _decode(fs)
        if img is None:
            continue
        images_used += 1
        dets = detect_fn(img)
        h, w = img.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            res = state.anomaly_detector.raw_score(cam, d["label"], crop)
            if res is None or res["raw_threshold"] <= 0:
                continue
            ratio = res["score"] / res["raw_threshold"]
            region_ratios.setdefault(d["label"], []).append(ratio)
            detections_total += 1

    return {
        "images_used": images_used,
        "detections_total": detections_total,
        "regions": region_ratios,
    }


def _summarize(region_ratios: dict) -> dict:
    """Return per-region stats + suggested multipliers."""
    out = {}
    for label, ratios in region_ratios.items():
        arr = np.array(ratios, dtype=np.float32)
        max_ratio = float(arr.max())
        out[label] = {
            "count": int(arr.size),
            "max_ratio": max_ratio,
            "mean_ratio": float(arr.mean()),
            "std_ratio": float(arr.std()),
            "suggested_multiplier": round(max_ratio * SAFETY_MARGIN, 3),
        }
    return out


@bp.route("/calibrate", methods=["POST"])
def calibrate():
    if state.detector is None or state.anomaly_detector is None:
        return jsonify({"success": False, "error": "Detector or anomaly detector not initialized"}), 500

    cam1_files = request.files.getlist("cam1")
    cam2_files = request.files.getlist("cam2")
    if not cam1_files and not cam2_files:
        return jsonify({"success": False, "error": "No images provided"}), 400

    t0 = time.time()
    cam1 = _gather_ratios("cam1", cam1_files, state.detector.detect_cam1) if cam1_files else None
    cam2 = _gather_ratios("cam2", cam2_files, state.detector.detect_cam2) if cam2_files else None
    elapsed = time.time() - t0

    cam1_summary = _summarize(cam1["regions"]) if cam1 else {}
    cam2_summary = _summarize(cam2["regions"]) if cam2 else {}

    # Suggested global multiplier = max ratio across everything * margin
    all_max = []
    for s in (cam1_summary, cam2_summary):
        for v in s.values():
            all_max.append(v["max_ratio"])
    suggested_global = round(max(all_max) * SAFETY_MARGIN, 3) if all_max else 1.0

    log.info("/calibrate cam1_imgs=%d cam2_imgs=%d total_dets=%d elapsed=%.1fs suggested_global=%.3f",
             cam1["images_used"] if cam1 else 0,
             cam2["images_used"] if cam2 else 0,
             (cam1["detections_total"] if cam1 else 0) + (cam2["detections_total"] if cam2 else 0),
             elapsed, suggested_global)

    return jsonify({
        "success": True,
        "elapsed_ms": int(elapsed * 1000),
        "safety_margin": SAFETY_MARGIN,
        "cam1": {
            "images_used":      cam1["images_used"]      if cam1 else 0,
            "detections_total": cam1["detections_total"] if cam1 else 0,
            "regions":          cam1_summary,
        },
        "cam2": {
            "images_used":      cam2["images_used"]      if cam2 else 0,
            "detections_total": cam2["detections_total"] if cam2 else 0,
            "regions":          cam2_summary,
        },
        "suggested_global_multiplier": suggested_global,
        "current_global_multiplier":   state.anomaly_detector.global_multiplier,
        "current_per_region":          state.anomaly_detector.per_region,
    })


@bp.route("/calibrate/save", methods=["POST"])
def save_calibration():
    if state.anomaly_detector is None:
        return jsonify({"success": False, "error": "Anomaly detector not initialized"}), 500

    data = request.get_json(silent=True) or {}
    global_multiplier = data.get("global_multiplier")
    per_region = data.get("per_region", {})

    if global_multiplier is None:
        return jsonify({"success": False, "error": "global_multiplier required"}), 400

    state.anomaly_detector.save_calibration(float(global_multiplier), per_region)
    log.info("/calibrate/save global=%.3f per_region_overrides=%d",
             float(global_multiplier),
             sum(len(v) for v in per_region.values()))
    return jsonify({
        "success": True,
        "global_multiplier": state.anomaly_detector.global_multiplier,
        "per_region": state.anomaly_detector.per_region,
    })
