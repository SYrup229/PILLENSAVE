"""YOLOv8 dual-camera region detector."""
import logging
from pathlib import Path
import numpy as np

from config import (
    BOX_LABELS, INFERENCE_IMG_SIZE,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
)

log = logging.getLogger("yolo_app.detector")


class YoloDualDetector:
    """Loads two YOLOv8 models (one per camera) and runs region detection."""

    def __init__(self, cam1_model_path: str, cam2_model_path: str, device: str = None):
        from ultralytics import YOLO  # heavy import — only when constructed

        self.cam1_model = YOLO(str(cam1_model_path)) if cam1_model_path else None
        self.cam2_model = YOLO(str(cam2_model_path)) if cam2_model_path else None

        # Track which device each model loaded onto (auto-selected by ultralytics)
        self.device = device or self._infer_device()

        if self.cam1_model:
            self.cam1_classes = self.cam1_model.names  # {id: name}
        if self.cam2_model:
            self.cam2_classes = self.cam2_model.names

    @staticmethod
    def _infer_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _run(self, model, img: np.ndarray, names: dict) -> list:
        """Run inference and return a list of detection dicts."""
        if model is None or img is None:
            return []

        results = model.predict(
            img,
            imgsz=INFERENCE_IMG_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            device=self.device,
        )
        log.info("inference: img_shape=%s imgsz=%d conf>=%.2f raw_results=%d",
                 img.shape, INFERENCE_IMG_SIZE, CONFIDENCE_THRESHOLD,
                 len(results) if results else 0)
        if not results:
            return []

        r = results[0]
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            log.info("  no boxes returned by model")
            return out

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss  = r.boxes.cls.cpu().numpy().astype(int)
        log.info("  raw detections: %d  max_conf=%.3f", len(boxes), float(confs.max()) if len(confs) else 0)

        for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, clss):
            label = names.get(int(cid), f"class_{cid}")
            out.append({
                "class_id": int(cid),
                "label": label,
                "confidence": float(conf),
                "box": [int(x1), int(y1), int(x2), int(y2)],  # xyxy in pixels
            })
        return out

    def detect_cam1(self, img: np.ndarray) -> list:
        return self._enforce_unique(self._run(self.cam1_model, img, self.cam1_classes))

    def detect_cam2(self, img: np.ndarray) -> list:
        return self._enforce_unique(self._run(self.cam2_model, img, self.cam2_classes))

    @staticmethod
    def _enforce_unique(detections: list) -> list:
        """Layout is fixed — keep only the highest-confidence detection per class."""
        best = {}
        for d in detections:
            cid = d["class_id"]
            if cid not in best or d["confidence"] > best[cid]["confidence"]:
                best[cid] = d
        return sorted(best.values(), key=lambda d: d["class_id"])
