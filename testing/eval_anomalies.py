"""
End-to-End Pipeline Evaluation on Synthetic Anomalies
=======================================================
Runs the full YOLO + PatchCore pipeline against the synthetic missing-wire
images produced by datagather/generate_anomalies.py, then compares the
per-region predictions against the ground truth in anomalies.json.

A region is predicted as anomalous if EITHER:
  - YOLO does not detect it in the frame (the wires were removed, so this is
    a valid signal), OR
  - YOLO does detect it AND PatchCore flags it as anomaly.

A region is predicted as normal if YOLO detects it AND PatchCore passes.

Outputs (into testing/results/anomalies_<timestamp>/):
  - summary.md
  - <cam>_confusion_matrix.png
  - <cam>_per_region.png
  - <cam>_samples.png
  - <cam>_metrics.json

Usage:
    python eval_anomalies.py                # both cameras, default 12 samples
    python eval_anomalies.py --cam cam1
    python eval_anomalies.py --samples 16
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from _render import (
    render_annotated_figure,
    confusion_matrix_2x2, per_region_bars,
    MATCH_GREEN, FP_ORANGE, FN_BLUE, ANOMALY_RED,
)


# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).resolve().parent      # yolo_app/testing
APP_ROOT      = SCRIPT_DIR.parent                    # yolo_app
RESULTS_DIR   = SCRIPT_DIR / "results"

ANOM_ROOT     = SCRIPT_DIR / "test_data" / "synthetic_anomalies"
CHECKPOINTS   = APP_ROOT / "checkpoints"
CAMERAS       = ["cam1", "cam2"]

# Make yolo_app's modules importable so we reuse the production inference code
sys.path.insert(0, str(APP_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_anomalies_json(cam_dir: Path) -> dict:
    p = cam_dir / "anomalies.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_camera(cam: str, detector, anomaly, out_dir: Path, n_samples: int = 0):
    cam_anom_dir = ANOM_ROOT / cam
    img_dir      = cam_anom_dir / "images"
    metadata     = load_anomalies_json(cam_anom_dir)
    if not metadata:
        print(f"[{cam}] no anomalies.json found at {cam_anom_dir}, skipping")
        return None

    image_files = sorted(img_dir.glob("*.jpg"))
    if not image_files:
        print(f"[{cam}] no images found at {img_dir}, skipping")
        return None

    detect_fn = detector.detect_cam1 if cam == "cam1" else detector.detect_cam2

    # Per-image results, used both for global metrics and for sample rendering
    per_image = []
    global_tp = global_fp = global_fn = global_tn = 0
    per_region_counts: dict = {}  # {region: {tp, fp, fn, tn}}

    for img_path in image_files:
        meta = metadata.get(img_path.name, {})
        missing_set = set(meta.get("missing_regions", []))
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        dets = detect_fn(img)
        # Run PatchCore on each detection
        detection_status = {}  # {label: ("ok" | "anomaly" | "no_bank")}
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            sc = anomaly.score(cam, d["label"], crop) if anomaly is not None else None
            if sc is None:
                detection_status[d["label"]] = "no_bank"
            elif sc["anomaly"]:
                detection_status[d["label"]] = "anomaly"
            else:
                detection_status[d["label"]] = "ok"

        # Determine the universe of regions to score: every region named in the
        # missing set + every region YOLO detected.
        all_regions = set(missing_set) | set(detection_status.keys())

        per_image_outcome = {"image": img_path.name,
                             "missing": sorted(missing_set),
                             "regions": {}}

        for region in all_regions:
            actually_anomalous = region in missing_set
            # Predicted anomalous if not detected, or detected and flagged
            detected = region in detection_status
            if not detected:
                predicted_anomalous = True
            else:
                predicted_anomalous = detection_status[region] == "anomaly"

            outcome = (
                "TP" if  actually_anomalous and  predicted_anomalous else
                "FN" if  actually_anomalous and not predicted_anomalous else
                "FP" if not actually_anomalous and  predicted_anomalous else
                "TN"
            )
            per_image_outcome["regions"][region] = {
                "actual": "anomaly" if actually_anomalous else "normal",
                "predicted": "anomaly" if predicted_anomalous else "normal",
                "outcome": outcome,
                "yolo_detected": detected,
                "detection_status": detection_status.get(region, "missing"),
            }

            if outcome == "TP": global_tp += 1
            elif outcome == "FP": global_fp += 1
            elif outcome == "FN": global_fn += 1
            else: global_tn += 1

            rc = per_region_counts.setdefault(region, {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
            rc[outcome] += 1

        per_image.append(per_image_outcome)

    # Aggregate per-region rates
    per_region = {}
    for region, c in per_region_counts.items():
        tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        per_region[region] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if (p + r) else 0.0,
        }

    total = global_tp + global_fp + global_fn + global_tn
    p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) else 0.0
    r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) else 0.0
    summary = {
        "cam": cam,
        "images_evaluated": len(per_image),
        "total_decisions": total,
        "tp": global_tp, "fp": global_fp, "fn": global_fn, "tn": global_tn,
        "precision": p,
        "recall":    r,
        "f1":        2 * p * r / (p + r) if (p + r) else 0.0,
        "accuracy":  (global_tp + global_tn) / total if total else 0.0,
        "per_region": per_region,
    }

    # Confusion matrix and per-region bars (shared helpers)
    cv2.imwrite(str(out_dir / f"{cam}_anomaly_cm.png"),
                confusion_matrix_2x2(global_tp, global_fp, global_fn, global_tn,
                                     f"{cam} - anomaly confusion matrix"))
    cv2.imwrite(str(out_dir / f"{cam}_per_region.png"),
                per_region_bars(per_region,
                                f"{cam} - per-region recall (green) and precision (blue)"))

    # Categorized sample figures: best TP, best FP, best FN
    sample_files = render_categorized_samples(
        cam, img_dir, per_image, detect_fn, anomaly, out_dir,
    )
    summary["sample_files"] = sample_files

    (out_dir / f"{cam}_per_image.json").write_text(json.dumps(per_image, indent=2))
    (out_dir / f"{cam}_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def _render_one(cam: str, img_path: Path, outcome: dict, detect_fn, anomaly,
                kind: str, out_dir: Path) -> str | None:
    """Render one full-size annotated figure. Returns saved filename or None."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    dets = detect_fn(img)
    items = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        crop = img[y1c:y2c, x1c:x2c]
        sc = anomaly.score(cam, d["label"], crop) if anomaly else None
        info = outcome["regions"].get(d["label"])
        outcome_label = info["outcome"] if info else "?"
        color = (ANOMALY_RED if outcome_label == "TP" else
                 FP_ORANGE   if outcome_label == "FP" else
                 MATCH_GREEN if outcome_label == "TN" else
                 FN_BLUE     if outcome_label == "FN" else
                 (180, 180, 180))
        extra = f"score {sc['score']:.2f} / thr {sc['threshold']:.2f}" if sc else "no bank"
        items.append({
            "box": (x1, y1, x2, y2),
            "label": d["label"],
            "color": color,
            "outcome": outcome_label,
            "extra": extra,
        })

    # Append "FN with no YOLO detection" as virtual items so the legend shows them
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for v in outcome["regions"].values():
        counts[v["outcome"]] += 1
    fn_no_det = [region for region, info in outcome["regions"].items()
                 if info["actual"] == "anomaly"
                 and info["outcome"] == "FN"
                 and not info["yolo_detected"]]
    title = (f"{cam}  -  {kind}  -  {img_path.name}  "
             f"-  TP={counts['TP']} FP={counts['FP']} "
             f"FN={counts['FN']} TN={counts['TN']}")

    extra_note = None
    if fn_no_det:
        extra_note = ("FN with no YOLO detection (region was removed, YOLO did not"
                      " detect anything there): " + ", ".join(fn_no_det))

    figure = render_annotated_figure(img, items, title, extra_note=extra_note)

    fname = f"{cam}_sample_{kind}.png"
    cv2.imwrite(str(out_dir / fname), figure)
    return fname


def render_categorized_samples(cam: str, img_dir: Path, per_image: list,
                               detect_fn, anomaly, out_dir: Path) -> list[str]:
    """Pick the best example of each failure mode + one clean pass and render them."""
    def counts(o):
        c = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for v in o["regions"].values():
            c[v["outcome"]] += 1
        return c

    annotated = [(o, counts(o)) for o in per_image]
    saved = []

    # Best TP example: most TPs, fewest FP/FN
    if any(c["TP"] for _, c in annotated):
        best_tp = max(annotated, key=lambda x: (x[1]["TP"], -x[1]["FP"] - x[1]["FN"]))
        f = _render_one(cam, img_dir / best_tp[0]["image"], best_tp[0],
                        detect_fn, anomaly, "true_positive", out_dir)
        if f: saved.append(f)

    # Worst FP example
    if any(c["FP"] for _, c in annotated):
        worst_fp = max(annotated, key=lambda x: x[1]["FP"])
        f = _render_one(cam, img_dir / worst_fp[0]["image"], worst_fp[0],
                        detect_fn, anomaly, "false_positive", out_dir)
        if f: saved.append(f)

    # Worst FN example
    if any(c["FN"] for _, c in annotated):
        worst_fn = max(annotated, key=lambda x: x[1]["FN"])
        f = _render_one(cam, img_dir / worst_fn[0]["image"], worst_fn[0],
                        detect_fn, anomaly, "false_negative", out_dir)
        if f: saved.append(f)

    # Clean pass example (no errors)
    clean = [(o, c) for o, c in annotated if c["FP"] == 0 and c["FN"] == 0]
    if clean:
        # Pick the one with the most TNs (clear-cut success)
        clean.sort(key=lambda x: -x[1]["TN"])
        f = _render_one(cam, img_dir / clean[0][0]["image"], clean[0][0],
                        detect_fn, anomaly, "clean_pass", out_dir)
        if f: saved.append(f)

    return saved


# ── Markdown report ──────────────────────────────────────────────────────────

def write_markdown(out_dir: Path, summaries: list):
    md = ["# Synthetic-Anomaly Pipeline Evaluation", ""]
    md.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_  \n")
    md.append("")
    md.append("Per-region decisions are aggregated across all evaluated images.")
    md.append("A region is predicted *anomaly* if YOLO did not detect it, or YOLO detected it and PatchCore flagged it.")
    md.append("")
    md.append("## Headline metrics")
    md.append("")
    md.append("| Camera | Images | Decisions | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        md.append(
            f"| {s['cam']} | {s['images_evaluated']} | {s['total_decisions']} "
            f"| {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['accuracy']:.3f} "
            f"| {s['tp']} | {s['fp']} | {s['fn']} | {s['tn']} |"
        )
    md.append("")
    for s in summaries:
        md.append(f"## {s['cam']}")
        md.append("")
        md.append(f"![Confusion matrix]({s['cam']}_anomaly_cm.png)")
        md.append("")
        md.append(f"![Per-region recall and precision]({s['cam']}_per_region.png)")
        md.append("")
        for fname in s.get("sample_files", []):
            md.append(f"![Sample]({fname})")
            md.append("")
        md.append("### Per-region details")
        md.append("")
        md.append("| Region | TP | FP | FN | TN | Precision | Recall | F1 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for region in sorted(s["per_region"].keys()):
            r = s["per_region"][region]
            md.append(f"| {region} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} "
                      f"| {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} |")
        md.append("")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate full pipeline on synthetic anomalies")
    parser.add_argument("--cam", choices=CAMERAS, default=None)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"anomalies_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}")

    print("Loading YOLO + PatchCore...")
    from models.detector import YoloDualDetector
    from models.anomaly  import PatchCoreDetector

    cam1_pt  = CHECKPOINTS / "cam1.pt"
    cam2_pt  = CHECKPOINTS / "cam2.pt"
    cam1_pkl = CHECKPOINTS / "cam1_patchcore.pkl"
    cam2_pkl = CHECKPOINTS / "cam2_patchcore.pkl"

    detector = YoloDualDetector(str(cam1_pt) if cam1_pt.exists() else None,
                                str(cam2_pt) if cam2_pt.exists() else None)
    anomaly  = PatchCoreDetector(str(cam1_pkl) if cam1_pkl.exists() else None,
                                 str(cam2_pkl) if cam2_pkl.exists() else None)

    print(f"  detector device: {detector.device}")
    print(f"  anomaly cam1: {anomaly.has_cam1}  cam2: {anomaly.has_cam2}")

    cams = [args.cam] if args.cam else CAMERAS
    summaries = []
    for cam in cams:
        print(f"\n[{cam}] evaluating...")
        t0 = time.time()
        s = evaluate_camera(cam, detector, anomaly, out_dir)
        if s:
            summaries.append(s)
            print(f"[{cam}] {s['images_evaluated']} images, "
                  f"P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
                  f"(took {time.time() - t0:.1f}s)")

    write_markdown(out_dir, summaries)
    print(f"\nReport: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
