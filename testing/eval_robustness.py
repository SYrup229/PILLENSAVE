"""
Robustness Evaluation — accuracy under elastic augmentation
=============================================================
Takes images from a user-specified folder, applies N random whole-image
shifts (using the same translation as elastic_augment.py), runs the full
YOLO + PatchCore pipeline on every variant, and reports three metrics:

  1. YOLO accuracy        — per-region precision / recall against ground truth
  2. PatchCore FPR        — rate at which normal regions are flagged anomaly
                            (assumes all source images are normal, i.e. all
                             PatchCore "anomaly" outputs are false positives)
  3. Combined accuracy    — fraction of variants where every detected region
                            was correctly classified (no YOLO miss, no PC FP)

Ground truth comes from datagather/timed_data/annotations_<cam>.json when
the source filename is found there; otherwise YOLO's prediction on the
unshifted image is used as pseudo-ground-truth.

Usage:
    python eval_robustness.py --source-dir <folder> --cam cam1
    python eval_robustness.py --source-dir <folder> --cam cam2 --variants 20 --num-images 100
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Adjustable parameters ────────────────────────────────────────────────────

DEFAULT_VARIANTS    = 10     # how many random shifts per source image (CLI override: --variants)
IOU_MATCH_THRESHOLD = 0.5    # min IoU to count a YOLO prediction as matching ground truth
MIN_SHIFT_PX        = 5      # min Euclidean shift magnitude
MAX_SHIFT_PX        = 50     # max Euclidean shift magnitude

# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent      # yolo_app/testing
APP_ROOT     = SCRIPT_DIR.parent                    # yolo_app
RESULTS_DIR  = SCRIPT_DIR / "results"
CHECKPOINTS  = APP_ROOT / "checkpoints"

sys.path.insert(0, str(APP_ROOT))


# ── Inlined elastic_augment helpers (no datagather dependency) ───────────────

def random_shift(rng: random.Random) -> tuple[int, int]:
    """Pick a random (dx, dy) with Euclidean magnitude in [MIN_SHIFT_PX, MAX_SHIFT_PX]."""
    mag = rng.uniform(MIN_SHIFT_PX, MAX_SHIFT_PX)
    angle = rng.uniform(0, 2 * math.pi)
    return int(round(mag * math.cos(angle))), int(round(mag * math.sin(angle)))


def translate_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate the whole image by (dx, dy), reflecting at edges."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_annotations(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / (((ax2 - ax1) * (ay2 - ay1)) +
                    ((bx2 - bx1) * (by2 - by1)) - inter)


def shift_gt_boxes(gt_dict: dict, dx: int, dy: int, w: int, h: int) -> list:
    """Apply (dx, dy) translation to each GT box and clip to image bounds.
    Returns list of (label, x1, y1, x2, y2)."""
    out = []
    for label, box in gt_dict.items():
        x1, y1, x2, y2 = [int(v) for v in box]
        nx1, ny1 = max(0, x1 + dx), max(0, y1 + dy)
        nx2, ny2 = min(w, x2 + dx), min(h, y2 + dy)
        if nx2 - nx1 < 4 or ny2 - ny1 < 4:
            continue
        out.append((label, nx1, ny1, nx2, ny2))
    return out


# ── Per-variant evaluation ───────────────────────────────────────────────────

def evaluate_variant(img, gt_boxes, anomaly, cam, detect_fn) -> dict:
    h, w = img.shape[:2]
    preds = detect_fn(img)

    # Match predictions to ground truth (same class, IoU >= threshold)
    matched_gt = set()
    n_tp = 0
    for p in preds:
        best_iou, best_gi = 0.0, None
        for gi, (label, x1, y1, x2, y2) in enumerate(gt_boxes):
            if gi in matched_gt or label != p["label"]:
                continue
            i = iou(p["box"], (x1, y1, x2, y2))
            if i > best_iou:
                best_iou, best_gi = i, gi
        if best_gi is not None and best_iou >= IOU_MATCH_THRESHOLD:
            matched_gt.add(best_gi)
            n_tp += 1

    n_fp = len(preds) - n_tp
    n_fn = len(gt_boxes) - n_tp

    # PatchCore: every detected region should be PASS (all source images normal)
    pc_total = 0
    pc_fp    = 0
    pc_fp_regions = []
    for p in preds:
        x1, y1, x2, y2 = p["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        sc = anomaly.score(cam, p["label"], crop) if anomaly else None
        if sc is None:
            continue
        pc_total += 1
        if sc["anomaly"]:
            pc_fp += 1
            pc_fp_regions.append(p["label"])

    combined_ok = (n_fp == 0 and n_fn == 0 and pc_fp == 0)

    return {
        "yolo_tp":       n_tp,
        "yolo_fp":       n_fp,
        "yolo_fn":       n_fn,
        "yolo_recall":    n_tp / len(gt_boxes) if gt_boxes else 0.0,
        "yolo_precision": n_tp / len(preds)    if preds    else 0.0,
        "pc_total":      pc_total,
        "pc_fp":         pc_fp,
        "pc_fpr":        pc_fp / pc_total if pc_total else 0.0,
        "pc_fp_regions": pc_fp_regions,
        "combined_ok":   combined_ok,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(per_test: list) -> dict:
    if not per_test:
        return {}
    yolo_tp = sum(t["yolo_tp"] for t in per_test)
    yolo_fp = sum(t["yolo_fp"] for t in per_test)
    yolo_fn = sum(t["yolo_fn"] for t in per_test)
    pc_total = sum(t["pc_total"] for t in per_test)
    pc_fp    = sum(t["pc_fp"]    for t in per_test)
    combined_ok = sum(1 for t in per_test if t["combined_ok"])

    # Per-region FP frequency for diagnostics
    region_fp_count = {}
    for t in per_test:
        for r in t["pc_fp_regions"]:
            region_fp_count[r] = region_fp_count.get(r, 0) + 1

    return {
        "total_tests": len(per_test),
        "yolo": {
            "tp": yolo_tp, "fp": yolo_fp, "fn": yolo_fn,
            "precision": yolo_tp / (yolo_tp + yolo_fp) if (yolo_tp + yolo_fp) else 0.0,
            "recall":    yolo_tp / (yolo_tp + yolo_fn) if (yolo_tp + yolo_fn) else 0.0,
        },
        "patchcore": {
            "total_decisions": pc_total,
            "false_positives": pc_fp,
            "fpr":             pc_fp / pc_total if pc_total else 0.0,
            "per_region_fp_count": region_fp_count,
        },
        "combined": {
            "perfect_variants": combined_ok,
            "accuracy":         combined_ok / len(per_test),
        },
    }


# ── Figures ──────────────────────────────────────────────────────────────────

def make_distribution_figure(per_test: list, summary: dict, cam: str, out_path: Path):
    yolo_recalls = np.array([t["yolo_recall"] for t in per_test])
    pc_fprs      = np.array([t["pc_fpr"]      for t in per_test])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white", dpi=120)

    # YOLO recall histogram
    axes[0].hist(yolo_recalls, bins=20, range=(0, 1.05),
                 color="#4dbf6c", edgecolor="#222", linewidth=0.5)
    axes[0].axvline(yolo_recalls.mean(), color="#cc2222", linestyle="--",
                    linewidth=1.5, label=f"mean = {yolo_recalls.mean():.3f}")
    axes[0].set_title("YOLO recall per variant", fontweight="bold")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Number of variants")
    axes[0].set_xlim(-0.02, 1.05)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend()

    # PatchCore FPR histogram
    axes[1].hist(pc_fprs, bins=20, range=(0, 1.05),
                 color="#e08c4a", edgecolor="#222", linewidth=0.5)
    axes[1].axvline(pc_fprs.mean(), color="#cc2222", linestyle="--",
                    linewidth=1.5, label=f"mean = {pc_fprs.mean():.3f}")
    axes[1].set_title("PatchCore false-positive rate per variant", fontweight="bold")
    axes[1].set_xlabel("FPR (anomaly verdicts on normal regions)")
    axes[1].set_ylabel("Number of variants")
    axes[1].set_xlim(-0.02, 1.05)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].legend()

    # Combined system: pass/fail bar
    n_pass = summary["combined"]["perfect_variants"]
    n_fail = summary["total_tests"] - n_pass
    bars = axes[2].bar(["All correct", "Any error"], [n_pass, n_fail],
                       color=["#4dbf6c", "#cc4444"], edgecolor="#222", linewidth=0.5)
    axes[2].set_title(
        f"Combined stability   (accuracy = {summary['combined']['accuracy']:.3f})",
        fontweight="bold",
    )
    axes[2].set_ylabel("Number of variants")
    axes[2].grid(axis="y", linestyle=":", alpha=0.5)
    for bar, val in zip(bars, [n_pass, n_fail]):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(n_pass, n_fail) * 0.01,
                     str(val), ha="center", fontweight="bold")

    fig.suptitle(
        f"Robustness evaluation — {cam}  ({summary['total_tests']} variants)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_per_region_figure(summary: dict, cam: str, out_path: Path):
    region_fp = summary["patchcore"].get("per_region_fp_count", {})
    if not region_fp:
        return False
    labels = sorted(region_fp.keys(), key=lambda r: -region_fp[r])
    counts = [region_fp[r] for r in labels]

    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.35 * len(labels) + 1.5)),
                           facecolor="white", dpi=120)
    y = np.arange(len(labels))
    ax.barh(y, counts, color="#e08c4a", edgecolor="#222", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of false-positive flags across all variants")
    ax.set_title(f"{cam} — PatchCore false positives by region", fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for i, v in enumerate(counts):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ── Markdown report ──────────────────────────────────────────────────────────

def write_markdown(out_dir: Path, summary: dict, cam: str, source_dir: Path,
                   n_variants: int, n_images: int, gt_sources: dict,
                   has_region_fig: bool):
    md = ["# Robustness Evaluation", ""]
    md.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_  ")
    md.append("")
    md.append("## Configuration")
    md.append("")
    md.append(f"- Source folder: `{source_dir}`")
    md.append(f"- Camera: `{cam}`")
    md.append(f"- Source images evaluated: {n_images}")
    md.append(f"- Random variants per image: {n_variants}")
    md.append(f"- Total tests: {summary['total_tests']}")
    md.append(f"- Ground truth: {gt_sources['annotation']} from annotations, "
              f"{gt_sources['pseudo']} from YOLO-on-original (pseudo)")
    md.append("")
    md.append("## Headline metrics")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| YOLO precision | {summary['yolo']['precision']:.4f} |")
    md.append(f"| YOLO recall | {summary['yolo']['recall']:.4f} |")
    md.append(f"| YOLO true positives | {summary['yolo']['tp']} |")
    md.append(f"| YOLO false positives | {summary['yolo']['fp']} |")
    md.append(f"| YOLO false negatives | {summary['yolo']['fn']} |")
    md.append(f"| PatchCore false-positive rate | {summary['patchcore']['fpr']:.4f} |")
    md.append(f"| PatchCore total decisions | {summary['patchcore']['total_decisions']} |")
    md.append(f"| PatchCore false positives | {summary['patchcore']['false_positives']} |")
    md.append(f"| Combined system accuracy | {summary['combined']['accuracy']:.4f} |")
    md.append(f"| Variants with zero errors | {summary['combined']['perfect_variants']} of {summary['total_tests']} |")
    md.append("")
    md.append(f"![Robustness distributions]({cam}_distribution.png)")
    if has_region_fig:
        md.append("")
        md.append(f"![Per-region PatchCore false positives]({cam}_per_region_fp.png)")
    md.append("")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline robustness under elastic augmentation")
    parser.add_argument("--source-dir", required=True, type=str,
                        help="Folder containing source images")
    parser.add_argument("--cam", choices=["cam1", "cam2"], required=True,
                        help="Which camera's models to use")
    parser.add_argument("--variants", type=int, default=DEFAULT_VARIANTS,
                        help=f"Random variants per image (default: {DEFAULT_VARIANTS})")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Limit to N random source images (default: all)")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Optional path to annotations_<cam>.json for ground truth. "
                             "If omitted, YOLO prediction on the unshifted image is used as pseudo-GT.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Source folder not found: {source_dir}")
        return

    # Import app components (random_shift / translate_image are defined inline above)
    from models.detector  import YoloDualDetector
    from models.anomaly   import PatchCoreDetector

    cam = args.cam
    cam1_pt  = CHECKPOINTS / "cam1.pt"
    cam2_pt  = CHECKPOINTS / "cam2.pt"
    cam1_pkl = CHECKPOINTS / "cam1_patchcore.pkl"
    cam2_pkl = CHECKPOINTS / "cam2_patchcore.pkl"

    print("Loading models...")
    detector = YoloDualDetector(
        str(cam1_pt) if cam1_pt.exists() else None,
        str(cam2_pt) if cam2_pt.exists() else None,
    )
    anomaly = PatchCoreDetector(
        str(cam1_pkl) if cam1_pkl.exists() else None,
        str(cam2_pkl) if cam2_pkl.exists() else None,
    )
    detect_fn = detector.detect_cam1 if cam == "cam1" else detector.detect_cam2
    print(f"  YOLO {cam}: loaded   PatchCore {cam}: "
          f"{'loaded' if (cam in anomaly.cams) else 'NOT loaded'}")

    ann_path = Path(args.annotations) if args.annotations else None
    annotations = load_annotations(ann_path)
    if annotations:
        print(f"  Annotations: {len(annotations)} entries from {ann_path}")
    else:
        print("  Annotations: none provided; using YOLO-on-original as pseudo-GT for every image")

    all_imgs = sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.png"))
    if not all_imgs:
        print(f"No images found in {source_dir}")
        return

    rng_sample = random.Random(args.seed)
    if args.num_images and args.num_images < len(all_imgs):
        all_imgs = rng_sample.sample(all_imgs, args.num_images)

    print(f"\nSource: {source_dir}")
    print(f"Camera: {cam}")
    print(f"Source images: {len(all_imgs)}")
    print(f"Variants per image: {args.variants}")
    print(f"Total tests: {len(all_imgs) * args.variants}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"robustness_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}\n")

    rng_shift = random.Random(args.seed)
    per_test  = []
    gt_sources = {"annotation": 0, "pseudo": 0}

    t0 = time.time()
    for img_idx, img_path in enumerate(all_imgs, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Determine ground truth
        ann = annotations.get(img_path.name)
        if ann is not None:
            gt_dict = {label: tuple(box) for label, box in ann["boxes"].items()
                       if isinstance(box, list)}
            gt_sources["annotation"] += 1
        else:
            preds_orig = detect_fn(img)
            gt_dict = {p["label"]: tuple(p["box"]) for p in preds_orig}
            gt_sources["pseudo"] += 1

        if not gt_dict:
            continue

        for v in range(args.variants):
            if v == 0:
                dx, dy = 0, 0
                variant_img = img
            else:
                dx, dy = random_shift(rng_shift)
                variant_img = translate_image(img, dx, dy)

            gt_boxes_shifted = shift_gt_boxes(gt_dict, dx, dy, w, h)
            result = evaluate_variant(variant_img, gt_boxes_shifted, anomaly,
                                      cam, detect_fn)
            result["image"]   = img_path.name
            result["variant"] = v
            result["dx"]      = dx
            result["dy"]      = dy
            per_test.append(result)

        if img_idx % 10 == 0 or img_idx == len(all_imgs):
            elapsed = time.time() - t0
            eta = elapsed / img_idx * (len(all_imgs) - img_idx)
            print(f"  [{img_idx}/{len(all_imgs)}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

    summary = aggregate(per_test)
    (out_dir / "per_test.json").write_text(json.dumps(per_test, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nRendering figures...")
    make_distribution_figure(per_test, summary, cam, out_dir / f"{cam}_distribution.png")
    has_region_fig = make_per_region_figure(summary, cam, out_dir / f"{cam}_per_region_fp.png")

    write_markdown(out_dir, summary, cam, source_dir,
                   args.variants, len(all_imgs), gt_sources, has_region_fig)

    print("\n=== Headline ===")
    print(f"  YOLO precision : {summary['yolo']['precision']:.4f}")
    print(f"  YOLO recall    : {summary['yolo']['recall']:.4f}")
    print(f"  PatchCore FPR  : {summary['patchcore']['fpr']:.4f}")
    print(f"  Combined acc.  : {summary['combined']['accuracy']:.4f} "
          f"({summary['combined']['perfect_variants']}/{summary['total_tests']})")
    print(f"\nReport: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
