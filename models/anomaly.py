"""PatchCore per-region anomaly detector. Mirrors the architecture in
enhanced_data/train_patchcore.py."""
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import ANOMALY_THRESHOLD_MULTIPLIER, CHECKPOINTS_DIR

CALIBRATION_PATH = CHECKPOINTS_DIR / "calibration.json"


class _FeatureExtractor(nn.Module):
    """ResNet18 layer2+layer3 features, concatenated and upsampled. Same as training."""

    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem_to_l2 = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2,
        )
        self.layer3 = net.layer3
        for p in self.parameters():
            p.requires_grad = False
        self.feature_dim = 128 + 256

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2 = self.stem_to_l2(x)
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)


class PatchCoreDetector:
    """Loads cam1 + cam2 PatchCore .pkl files and scores cropped regions."""

    def __init__(self, cam1_pkl: str | None, cam2_pkl: str | None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extractor = _FeatureExtractor().to(self.device).eval()

        self.cams = {}
        if cam1_pkl and Path(cam1_pkl).exists():
            self.cams["cam1"] = self._load(cam1_pkl)
        if cam2_pkl and Path(cam2_pkl).exists():
            self.cams["cam2"] = self._load(cam2_pkl)

        # Calibration overrides
        self.global_multiplier = float(ANOMALY_THRESHOLD_MULTIPLIER)
        self.per_region: dict[str, dict[str, float]] = {"cam1": {}, "cam2": {}}
        self._load_calibration()

    def _load_calibration(self):
        if not CALIBRATION_PATH.exists():
            return
        try:
            with open(CALIBRATION_PATH) as f:
                data = json.load(f)
            self.global_multiplier = float(data.get("global_multiplier", self.global_multiplier))
            for cam in ("cam1", "cam2"):
                self.per_region[cam] = {k: float(v) for k, v in data.get("per_region", {}).get(cam, {}).items()}
        except Exception as e:
            print(f"[anomaly] Failed to load calibration: {e}")

    def save_calibration(self, global_multiplier: float, per_region: dict):
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_multiplier": float(global_multiplier),
            "per_region": {cam: {k: float(v) for k, v in d.items()} for cam, d in per_region.items()},
        }
        with open(CALIBRATION_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        self.global_multiplier = float(global_multiplier)
        self.per_region = {cam: dict(d) for cam, d in per_region.items()}

    def _multiplier_for(self, cam: str, label: str) -> float:
        return self.per_region.get(cam, {}).get(label, self.global_multiplier)

    def _load(self, path: str) -> dict:
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Move banks to GPU once for fast scoring
        banks = {
            label: torch.from_numpy(b).to(self.device)
            for label, b in data["region_banks"].items()
        }
        return {
            "banks": banks,
            "thresholds": data["region_thresholds"],
            "k": int(data["k_nearest"]),
            "mean": np.array(data["imagenet_mean"], dtype=np.float32),
            "std":  np.array(data["imagenet_std"],  dtype=np.float32),
        }

    @property
    def has_cam1(self) -> bool:
        return "cam1" in self.cams

    @property
    def has_cam2(self) -> bool:
        return "cam2" in self.cams

    def region_count(self, cam: str) -> int:
        return len(self.cams[cam]["banks"]) if cam in self.cams else 0

    # ── inference ─────────────────────────────────────────────────────────

    @staticmethod
    def _pad_min16(crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        pad_h = max(0, 16 - h)
        pad_w = max(0, 16 - w)
        if pad_h or pad_w:
            crop_bgr = cv2.copyMakeBorder(crop_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        return crop_bgr

    def _preprocess(self, crop_bgr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
        crop_bgr = self._pad_min16(crop_bgr)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - mean) / std
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
        return tensor.to(self.device, non_blocking=True)

    def raw_score(self, cam: str, label: str, crop_bgr: np.ndarray) -> dict | None:
        """Returns {score, raw_threshold} without applying any multiplier. Used by /calibrate."""
        if cam not in self.cams:
            return None
        model = self.cams[cam]
        if label not in model["banks"]:
            return None
        bank = model["banks"][label]
        if bank.shape[0] == 0:
            return None

        tensor = self._preprocess(crop_bgr, model["mean"], model["std"])
        feat = self.extractor(tensor)
        _, C, _, _ = feat.shape
        q = feat.permute(0, 2, 3, 1).reshape(-1, C)

        d = torch.cdist(q, bank)
        k_eff = min(model["k"], bank.shape[0])
        topk, _ = torch.topk(d, k_eff, largest=False)
        per_patch_score = topk.mean(dim=1)
        score = float(per_patch_score.max().item())

        return {
            "score": score,
            "raw_threshold": float(model["thresholds"][label]),
        }

    def score(self, cam: str, label: str, crop_bgr: np.ndarray) -> dict | None:
        """Returns {score, threshold, anomaly} with the active multiplier applied."""
        raw = self.raw_score(cam, label, crop_bgr)
        if raw is None:
            return None
        threshold = raw["raw_threshold"] * self._multiplier_for(cam, label)
        return {
            "score": raw["score"],
            "threshold": threshold,
            "anomaly": raw["score"] > threshold,
        }
