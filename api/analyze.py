"""Capture + run YOLO inference on both cameras."""
import logging
import time

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

import state
from config import BOX_LABELS
from utils import encode_image_to_base64

bp = Blueprint("analyze", __name__)
log = logging.getLogger("yolo_app.analyze")


@bp.route("/analyze", methods=["POST"])
def analyze():
    if state.camera_manager is None:
        return jsonify({"success": False, "error": "Cameras not initialized"}), 500
    if state.detector is None:
        return jsonify({"success": False, "error": "Detector not initialized"}), 500

    t0 = time.time()
    cap = state.camera_manager.capture()
    if not cap["success"]:
        return jsonify({"success": False, "error": "Camera capture failed", "errors": cap["errors"]}), 500
    t_capture = time.time() - t0

    t1 = time.time()
    cam1_dets = state.detector.detect_cam1(cap["cam1_frame"]) if cap["cam1_frame"] is not None else []
    cam2_dets = state.detector.detect_cam2(cap["cam2_frame"]) if cap["cam2_frame"] is not None else []
    t_infer = time.time() - t1

    t2 = time.time()
    score_detections("cam1", cap["cam1_frame"], cam1_dets)
    score_detections("cam2", cap["cam2_frame"], cam2_dets)
    t_anom = time.time() - t2

    summary = build_summary(cam1_dets, cam2_dets, t_capture, t_infer, t_anom)
    result = {
        "success": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cam1_image": cap["cam1_b64"],
        "cam2_image": cap["cam2_b64"],
        "cam1_detections": cam1_dets,
        "cam2_detections": cam2_dets,
        "summary": summary,
    }
    state.last_result = result
    log.info("/analyze cam1=%d cam2=%d capture=%dms infer=%dms anomaly=%dms anomalies=%d",
             summary["cam1_count"], summary["cam2_count"],
             summary["capture_ms"], summary["inference_ms"], summary["anomaly_ms"],
             summary["anomaly_count"])
    return jsonify(result)


def score_detections(cam: str, frame, detections: list):
    """Run PatchCore on each detection's crop. Mutates the detection dicts in place."""
    if state.anomaly_detector is None or frame is None:
        for d in detections:
            d["status"] = "detected"
            d["anomaly"] = None
            d["score"] = None
            d["threshold"] = None
        return

    h, w = frame.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            d["status"] = "invalid"
            d["score"] = None
            d["threshold"] = None
            d["anomaly"] = None
            continue

        crop = frame[y1:y2, x1:x2]
        result = state.anomaly_detector.score(cam, d["label"], crop)
        if result is None:
            d["status"] = "no_bank"
            d["score"] = None
            d["threshold"] = None
            d["anomaly"] = None
        else:
            d["score"] = result["score"]
            d["threshold"] = result["threshold"]
            d["anomaly"] = result["anomaly"]
            d["status"] = "anomaly" if result["anomaly"] else "ok"


def build_summary(cam1_dets, cam2_dets, t_capture, t_infer, t_anom):
    anomaly_count = sum(1 for d in cam1_dets + cam2_dets if d.get("anomaly") is True)
    return {
        "cam1_count":      len(cam1_dets),
        "cam2_count":      len(cam2_dets),
        "total_count":     len(cam1_dets) + len(cam2_dets),
        "expected_count":  len(BOX_LABELS) * 2,
        "anomaly_count":   anomaly_count,
        "capture_ms":      int(t_capture * 1000),
        "inference_ms":    int(t_infer * 1000),
        "anomaly_ms":      int(t_anom * 1000),
        "stage2_loaded":   state.anomaly_detector is not None,
    }


def _file_to_image(file_storage):
    if file_storage is None:
        return None
    buf = np.frombuffer(file_storage.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@bp.route("/analyze-upload", methods=["POST"])
def analyze_upload():
    if state.detector is None:
        return jsonify({"success": False, "error": "Detector not initialized"}), 500

    cam1_img = _file_to_image(request.files.get("cam1"))
    cam2_img = _file_to_image(request.files.get("cam2"))
    if cam1_img is None and cam2_img is None:
        return jsonify({"success": False, "error": "No images provided"}), 400

    t1 = time.time()
    cam1_dets = state.detector.detect_cam1(cam1_img) if cam1_img is not None else []
    cam2_dets = state.detector.detect_cam2(cam2_img) if cam2_img is not None else []
    t_infer = time.time() - t1

    t2 = time.time()
    score_detections("cam1", cam1_img, cam1_dets)
    score_detections("cam2", cam2_img, cam2_dets)
    t_anom = time.time() - t2

    summary = build_summary(cam1_dets, cam2_dets, 0.0, t_infer, t_anom)
    result = {
        "success": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S") + " (upload)",
        "cam1_image": encode_image_to_base64(cam1_img) if cam1_img is not None else None,
        "cam2_image": encode_image_to_base64(cam2_img) if cam2_img is not None else None,
        "cam1_detections": cam1_dets,
        "cam2_detections": cam2_dets,
        "summary": summary,
    }
    state.last_result = result
    log.info("/analyze-upload cam1=%d cam2=%d infer=%dms anomaly=%dms anomalies=%d",
             summary["cam1_count"], summary["cam2_count"],
             summary["inference_ms"], summary["anomaly_ms"], summary["anomaly_count"])
    return jsonify(result)
