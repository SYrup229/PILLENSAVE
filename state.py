"""Shared application state."""

camera_manager   = None     # models.camera.CameraManager
detector         = None     # models.detector.YoloDualDetector
anomaly_detector = None     # models.anomaly.PatchCoreDetector (optional)

# Last analysis result (for /history-style polling if needed later)
last_result = None
