"""Matplotlib-based rendering helpers for thesis-quality evaluation figures.

Each figure is composed top-to-bottom:
  1. Title bar
  2. Large image with color-coded numbered boxes (numbers drawn INSIDE the box)
  3. Compact data table with one row per detection, colored to match its box

All renderers return a numpy BGR image (so the caller can `cv2.imwrite`).
"""

from __future__ import annotations

import io

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Thesis-friendly RGB colors (matplotlib wants RGB 0-1)
MATCH_GREEN  = (0.30, 0.75, 0.40)
GT_YELLOW    = (0.95, 0.78, 0.10)
FP_ORANGE    = (0.95, 0.55, 0.15)
FN_BLUE      = (0.20, 0.55, 0.85)
ANOMALY_RED  = (0.90, 0.25, 0.25)
PRIMARY_BLUE = (0.15, 0.50, 0.85)

# Aliases as BGR tuples (255 scale) for any cv2 callers that import from here
def _to_bgr255(rgb):
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

BGR_MATCH   = _to_bgr255(MATCH_GREEN)
BGR_GT      = _to_bgr255(GT_YELLOW)
BGR_FP      = _to_bgr255(FP_ORANGE)
BGR_FN      = _to_bgr255(FN_BLUE)
BGR_ANOMALY = _to_bgr255(ANOMALY_RED)
BGR_BLUE    = _to_bgr255(PRIMARY_BLUE)


def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _figure_to_bgr(fig) -> np.ndarray:
    """Render a matplotlib figure to a BGR numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ── Sample figure: image + table ─────────────────────────────────────────────

def render_annotated_figure(
    img_bgr: np.ndarray,
    items: list[dict],
    title: str,
    *,
    show_only_errors_in_table: bool = False,
    extra_note: str | None = None,
) -> np.ndarray:
    """Render image + table figure.

    Each dict in `items` must contain:
      - box: (x1, y1, x2, y2) in original image coordinates
      - label: region label
      - color: matplotlib RGB tuple (0-1)
      - outcome: short outcome string (e.g. "TP", "FP", "match")
      - extra: optional short suffix (e.g. "score 5.2 / thr 4.1")
    """
    img_rgb = _bgr_to_rgb(img_bgr)
    H, W = img_rgb.shape[:2]

    # Decide which rows go in the table
    if show_only_errors_in_table:
        table_rows = [(i + 1, it) for i, it in enumerate(items)
                      if it["outcome"] not in ("match", "TN", "ok")]
    else:
        table_rows = [(i + 1, it) for i, it in enumerate(items)]

    # Figure layout: title (small), image (large), table (compact), optional note
    n_table_rows = max(1, len(table_rows))
    table_h_inches = 0.35 * n_table_rows + 0.6
    note_h_inches = 0.5 if extra_note else 0.0
    image_h_inches = W * (6.0 / 12.0) * (H / W) * 0.45 + 6  # heuristic
    image_h_inches = max(5.5, min(image_h_inches, 9.0))

    fig_w = 14
    fig_h = 0.7 + image_h_inches + table_h_inches + note_h_inches

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=120)
    height_ratios = [0.5, image_h_inches, table_h_inches]
    if extra_note:
        height_ratios.append(note_h_inches)
    gs = fig.add_gridspec(len(height_ratios), 1,
                          height_ratios=height_ratios,
                          hspace=0.12, left=0.04, right=0.98,
                          top=0.98, bottom=0.02)

    # Title
    ax_t = fig.add_subplot(gs[0])
    ax_t.axis("off")
    ax_t.text(0.0, 0.5, title, fontsize=14, fontweight="bold",
              va="center", ha="left", family="DejaVu Sans")

    # Image
    ax_i = fig.add_subplot(gs[1])
    ax_i.imshow(img_rgb)
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    for spine in ax_i.spines.values():
        spine.set_visible(False)

    for i, it in enumerate(items, 1):
        x1, y1, x2, y2 = it["box"]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.4, edgecolor=it["color"], facecolor="none",
        )
        ax_i.add_patch(rect)
        # Large number INSIDE the top-left corner of the box
        ax_i.text(
            x1 + 4, y1 + 4, str(i),
            fontsize=14, fontweight="bold",
            ha="left", va="top",
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=it["color"],
                      edgecolor="black", linewidth=0.6),
        )

    # Table
    ax_tab = fig.add_subplot(gs[2])
    ax_tab.axis("off")
    if not table_rows:
        ax_tab.text(0.5, 0.5, "(no errors)", fontsize=11, color="#666666",
                    ha="center", va="center", style="italic")
    else:
        cell_text = []
        cell_colors = []
        for n, it in table_rows:
            row = [
                str(n),
                it["label"],
                it["outcome"],
                it.get("extra", "") or "",
            ]
            cell_text.append(row)
            tinted = tuple(0.85 + 0.15 * c for c in it["color"])
            cell_colors.append([tinted] * 4)
        table = ax_tab.table(
            cellText=cell_text,
            cellColours=cell_colors,
            colLabels=["#", "Region", "Outcome", "Score / threshold"],
            colWidths=[0.05, 0.32, 0.18, 0.45],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.25)
        for (r, _c), cell in table.get_celld().items():
            cell.set_edgecolor("#999999")
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_facecolor("#333344")
                cell.set_text_props(color="white", fontweight="bold")

    if extra_note:
        ax_n = fig.add_subplot(gs[3])
        ax_n.axis("off")
        ax_n.text(0.0, 0.5, extra_note, fontsize=10, color=FN_BLUE,
                  ha="left", va="center", style="italic", wrap=True)

    out = _figure_to_bgr(fig)
    plt.close(fig)
    return out


# ── Confusion matrix (2x2) ───────────────────────────────────────────────────

def confusion_matrix_2x2(tp: int, fp: int, fn: int, tn: int, title: str) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white", dpi=120)

    matrix = np.array([[tn, fp], [fn, tp]])
    names = np.array([["TN", "FP"], ["FN", "TP"]])
    colors_correct = "#cfe9d4"
    colors_wrong   = "#f7d8d4"
    color_grid = np.array([[colors_correct, colors_wrong],
                           [colors_wrong, colors_correct]])

    # Background squares
    for r in range(2):
        for c in range(2):
            ax.add_patch(mpatches.Rectangle((c, 1 - r), 1, 1,
                                            facecolor=color_grid[r, c],
                                            edgecolor="#888888", linewidth=0.8))

    total = max(1, tp + fp + fn + tn)
    for r in range(2):
        for c in range(2):
            v = matrix[r, c]
            ax.text(c + 0.5, 1 - r + 0.65, names[r, c],
                    fontsize=14, fontweight="bold", ha="center", va="center",
                    color="#444444")
            ax.text(c + 0.5, 1 - r + 0.45, f"{v}",
                    fontsize=28, fontweight="bold", ha="center", va="center",
                    color="#1a1a1a")
            ax.text(c + 0.5, 1 - r + 0.22, f"{100 * v / total:.1f}%",
                    fontsize=11, ha="center", va="center", color="#666666")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Normal", "Anomaly"], fontsize=11)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Anomaly", "Normal"], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold", labelpad=10)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    out = _figure_to_bgr(fig)
    plt.close(fig)
    return out


# ── Per-region bars ──────────────────────────────────────────────────────────

def per_region_bars(per_region: dict, title: str) -> np.ndarray:
    labels = sorted(per_region.keys())
    if not labels:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    n = len(labels)
    fig_h = max(3, 0.4 * n + 1.2)
    fig, ax = plt.subplots(figsize=(13, fig_h), facecolor="white", dpi=120)

    y = np.arange(n)
    recalls    = np.array([per_region[k]["recall"]    for k in labels])
    precisions = np.array([per_region[k]["precision"] for k in labels])

    ax.barh(y + 0.18, recalls,    height=0.36, color=MATCH_GREEN,
            label="Recall", edgecolor="#444444", linewidth=0.4)
    ax.barh(y - 0.18, precisions, height=0.36, color=PRIMARY_BLUE,
            label="Precision", edgecolor="#444444", linewidth=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True)

    # Numeric annotations at the end of each bar
    for i in range(len(labels)):
        ax.text(recalls[i] + 0.01,    i + 0.18, f"{recalls[i]:.2f}",
                fontsize=8, va="center", color="#333333")
        ax.text(precisions[i] + 0.01, i - 0.18, f"{precisions[i]:.2f}",
                fontsize=8, va="center", color="#333333")

    fig.tight_layout()
    out = _figure_to_bgr(fig)
    plt.close(fig)
    return out
