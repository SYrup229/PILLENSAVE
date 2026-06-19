"""Camera capture endpoints."""
import logging

from flask import Blueprint, jsonify

import state

bp = Blueprint("camera", __name__)
log = logging.getLogger("yolo_app.camera")


@bp.route("/capture", methods=["POST"])
def capture():
    if state.camera_manager is None:
        return jsonify({"success": False, "error": "Cameras not initialized"}), 500

    result = state.camera_manager.capture()
    log.info("/capture success=%s cam1=%s cam2=%s",
             result.get("success"),
             "ok" if result.get("cam1_b64") else "—",
             "ok" if result.get("cam2_b64") else "—")
    # Don't ship the numpy frames over JSON
    return jsonify({
        "success": result["success"],
        "cam1": result["cam1_b64"],
        "cam2": result["cam2_b64"],
        "errors": result["errors"],
    })


@bp.route("/camera-status")
def camera_status():
    if state.camera_manager is None:
        return jsonify({"cam1": False, "cam2": False})

    cm = state.camera_manager
    with cm.lock:
        if not (cm.cam1_cap and cm.cam1_cap.isOpened()):
            cm.cam1_cap = cm._open_camera(cm.cam1_index)
        if not (cm.cam2_cap and cm.cam2_cap.isOpened()):
            cm.cam2_cap = cm._open_camera(cm.cam2_index)

    return jsonify({
        "cam1": cm.cam1_cap is not None and cm.cam1_cap.isOpened(),
        "cam2": cm.cam2_cap is not None and cm.cam2_cap.isOpened(),
    })
