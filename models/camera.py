"""Dual-camera USB capture manager."""
import threading
import platform

import cv2

from utils import encode_image_to_base64
from config import CAMERA_WIDTH, CAMERA_HEIGHT


class CameraManager:
    """Manages two USB cameras (cam1 and cam2)."""

    def __init__(self, cam1_index: int, cam2_index: int):
        self.cam1_index = cam1_index
        self.cam2_index = cam2_index
        self.cam1_cap = None
        self.cam2_cap = None
        self.lock = threading.Lock()

    def _open_camera(self, index: int) -> cv2.VideoCapture:
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            for _ in range(5):
                cap.read()  # warm up
        return cap

    def open(self) -> dict:
        status = {"cam1": False, "cam2": False, "errors": []}
        self.cam1_cap = self._open_camera(self.cam1_index)
        if self.cam1_cap.isOpened():
            status["cam1"] = True
        else:
            status["errors"].append(f"cam1 (index {self.cam1_index}) failed to open")

        self.cam2_cap = self._open_camera(self.cam2_index)
        if self.cam2_cap.isOpened():
            status["cam2"] = True
        else:
            status["errors"].append(f"cam2 (index {self.cam2_index}) failed to open")
        return status

    def _try_capture(self, cap, index):
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return cap, frame
            cap.release()

        cap = self._open_camera(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return cap, frame
            cap.release()
        return None, None

    def capture(self) -> dict:
        """Returns {success, cam1_frame, cam2_frame, cam1_b64, cam2_b64, errors}."""
        with self.lock:
            result = {
                "success": False,
                "cam1_frame": None, "cam2_frame": None,
                "cam1_b64":   None, "cam2_b64":   None,
                "errors": [],
            }
            self.cam1_cap, f1 = self._try_capture(self.cam1_cap, self.cam1_index)
            if f1 is not None:
                result["cam1_frame"] = f1
                result["cam1_b64"]   = encode_image_to_base64(f1)
            else:
                result["errors"].append("cam1 not available")

            self.cam2_cap, f2 = self._try_capture(self.cam2_cap, self.cam2_index)
            if f2 is not None:
                result["cam2_frame"] = f2
                result["cam2_b64"]   = encode_image_to_base64(f2)
            else:
                result["errors"].append("cam2 not available")

            result["success"] = result["cam1_b64"] is not None or result["cam2_b64"] is not None
            return result

    def release(self):
        if self.cam1_cap: self.cam1_cap.release()
        if self.cam2_cap: self.cam2_cap.release()
