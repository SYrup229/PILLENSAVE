# PILLENSAVE — Wire Inspector

Dual-camera YOLOv8 + PatchCore inspection app for wiring connectors.

## Layout

```
yolo_app/
├── app.py                    Flask entry point
├── config.py                 Paths, thresholds, BOX_LABELS
├── state.py                  Shared runtime state
├── utils.py                  Image encode helpers
├── models/
│   ├── detector.py           YOLOv8 dual-camera inference
│   ├── anomaly.py            PatchCore inference + calibration
│   └── camera.py             Dual USB camera manager
├── api/
│   ├── camera.py             /capture, /camera-status
│   ├── analyze.py            /analyze, /analyze-upload
│   ├── calibrate.py          /calibrate, /calibrate/save
│   └── views.py              /
├── templates/index.html      Single-page UI (Live + Calibration tabs)
├── checkpoints/              Drop trained weights here (not in repo)
└── testing/                  Evaluation scripts
    ├── eval_anomalies.py
    ├── eval_robustness.py
    └── test_data/synthetic_anomalies/   25 missing-wire test images per camera
```

## Required model weights (not in repo)

Drop these in `checkpoints/`:

| File | Source |
|---|---|
| `cam1.pt`, `cam2.pt` | YOLOv8m trained per camera (Ultralytics `best.pt` renamed) |
| `cam1_patchcore.pkl`, `cam2_patchcore.pkl` | PatchCore memory banks per camera |
| `calibration.json` (optional) | Per-region threshold multipliers from the Calibration tab |

The PatchCore `.pkl` files are large (>800 MB each) and intentionally excluded
from this repo.

## Install

```bash
pip install flask ultralytics opencv-python torch torchvision numpy matplotlib
```

## Run

```bash
python app.py                                  # default USB cameras 0 and 1
python app.py --cam1-camera 0 --cam2-camera 2  # explicit indices
```

Open `http://<host>:5000`.

## Testing

```bash
# End-to-end pipeline on bundled synthetic missing-wire images
python testing/eval_anomalies.py

# Robustness sweep on any folder of normal-state images
python testing/eval_robustness.py --source-dir /path/to/images --cam cam1
```

Reports land in `testing/results/<timestamp>/`.
