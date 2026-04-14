"""Generate publication-ready figures from the current comparison artifacts."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.figure_style import add_adjusted_labels, apply_publication_style, save_figure, style_axes


apply_publication_style()

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_JSON = REPO_ROOT / "results" / "comparison_final" / "comparison" / "comparison_summary.json"
COMPARISON_DIR = COMPARISON_JSON.parent

METHOD_COLORS = {
    "mc_dropout": "#4C72B0",
    "tta": "#DD8452",
}
MODE_MARKERS = {
    "global": "o",
    "adaptive": "^",
    "conf_aware": "s",
}
DISPLAY_NAMES = {
    "mc_dropout": "MC + global",
    "mc_dropout_adaptive": "MC + adaptive",
    "mc_dropout_conf_aware": "MC + conf-aware",
    "tta": "TTA + global",
    "tta_adaptive": "TTA + adaptive",
    "tta_conf_aware": "TTA + conf-aware",
}


def _load_summary() -> dict:
    with open(COMPARISON_JSON) as f:
        return json.load(f)


def _err_reduction_pct(summary: dict) -> float:
    before = float(summary["adt_error_before"])
    after = float(summary["adt_error_after"])
    return 100.0 * (before - after) / max(before, 1e-12)


def make_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 5.6), constrained_layout=True)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.34", facecolor="#E8F0FE", edgecolor="#4285F4", linewidth=1.6)
    box_kw2 = dict(boxstyle="round,pad=0.34", facecolor="#FEF3E8", edgecolor="#EA8D2F", linewidth=1.6)
    box_kw3 = dict(boxstyle="round,pad=0.34", facecolor="#E8FEE8", edgecolor="#34A853", linewidth=1.6)
    box_kw_out = dict(boxstyle="round,pad=0.34", facecolor="#FCE4EC", edgecolor="#C62828", linewidth=1.6)

    ax.text(3.5, 4.72, "Stage 1: Prediction + Uncertainty", fontsize=13, fontweight="bold", ha="center", color="#2E5AAC")
    ax.text(9.5, 4.72, "Stage 2: Deferral Policy", fontsize=13, fontweight="bold", ha="center", color="#B46612")
    ax.text(12.65, 4.72, "Output", fontsize=13, fontweight="bold", ha="center", color="#9E1F3D")

    ax.text(
        0.82,
        2.5,
        "Retinal\nImage",
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="#F4F5F7", edgecolor="#5B6470", linewidth=1.4),
    )
    ax.text(2.8, 3.55, "U-Net\n(ResNet-34)", fontsize=9.4, ha="center", va="center", bbox=box_kw)
    ax.text(5.05, 3.55, "Prediction\nmap $\\hat{\\mathbf{p}}$", fontsize=9.4, ha="center", va="center", bbox=box_kw)
    ax.text(2.8, 1.45, "MC Dropout\n($T$=30 passes)\nor TTA ($K$=6)", fontsize=8.4, ha="center", va="center", bbox=box_kw)
    ax.text(5.05, 1.45, "Uncertainty\nmap $\\mathbf{u}$", fontsize=9.4, ha="center", va="center", bbox=box_kw)

    ax.text(7.9, 3.55, "Global\n$\\delta = \\mathbb{1}[u \\leq \\tau]$", fontsize=8.2, ha="center", va="center", bbox=box_kw2)
    ax.text(9.62, 2.5, "Adaptive\n$\\delta = \\mathbb{1}[u \\leq Q_\\alpha]$", fontsize=8.2, ha="center", va="center", bbox=box_kw2)
    ax.text(7.9, 1.45, "Conf-aware\n$s = u(1-c)$", fontsize=8.2, ha="center", va="center", bbox=box_kw2)

    ax.text(8.7, 3.0, "or", fontsize=9, ha="center", va="center", fontstyle="italic", color="#80858E")
    ax.text(8.7, 2.0, "or", fontsize=9, ha="center", va="center", fontstyle="italic", color="#80858E")

    ax.text(11.5, 2.5, "Accept /\nDefer\nmask", fontsize=9.2, ha="center", va="center", bbox=box_kw3)
    ax.text(13.0, 3.45, "Automated\nreport", fontsize=8.2, ha="center", va="center", bbox=box_kw_out)
    ax.text(13.0, 1.55, "Clinician\nreview", fontsize=8.2, ha="center", va="center", bbox=box_kw_out)

    arrow_kw = dict(arrowstyle="->", color="#39424E", lw=1.6, mutation_scale=15)
    dashed_kw = dict(arrowstyle="->", color="#59616C", lw=1.1, mutation_scale=12, linestyle="--")
    ax.annotate("", xy=(2.0, 3.5), xytext=(1.4, 2.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(2.0, 1.5), xytext=(1.4, 2.2), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.2, 3.5), xytext=(3.6, 3.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.2, 1.5), xytext=(3.6, 1.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.0, 3.5), xytext=(5.85, 3.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.0, 2.5), xytext=(5.85, 3.2), arrowprops=dashed_kw)
    ax.annotate("", xy=(7.0, 1.5), xytext=(5.85, 1.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.0, 2.5), xytext=(5.85, 1.8), arrowprops=dashed_kw)
    ax.annotate("", xy=(10.7, 2.5), xytext=(10.22, 2.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(12.3, 3.28), xytext=(12.02, 2.78), arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.6, mutation_scale=15))
    ax.annotate("", xy=(12.3, 1.72), xytext=(12.02, 2.22), arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.6, mutation_scale=15))
    ax.text(12.12, 3.1, "accept", fontsize=7.4, color="#2E8B57", fontstyle="italic", rotation=31)
    ax.text(12.12, 1.95, "defer", fontsize=7.4, color="#C62828", fontstyle="italic", rotation=-31)

    ax.text(
        7.0,
        0.32,
        "2 uncertainty methods × 3 deferral policies = 6 operating points evaluated",
        fontsize=10,
        ha="center",
        va="center",
        color="#59616C",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#F7F8FA", edgecolor="#D5D9E0"),
    )

    save_figure(fig, OUT_DIR / "pipeline.png")
    print("Generated: pipeline.png")


def make_summary_scatter() -> None:
    summary = _load_summary()
    order = [
        "mc_dropout",
        "mc_dropout_adaptive",
        "mc_dropout_conf_aware",
        "tta",
        "tta_adaptive",
        "tta_conf_aware",
    ]
    configs = []
    for key in order:
        row = summary.get(key)
        if row is None or row.get("calibration") == "temperature":
            continue
        method = "tta" if key.startswith("tta") else "mc_dropout"
        mode = "global"
        if "adaptive" in key and "conf" not in key:
            mode = "adaptive"
        elif "conf_aware" in key:
            mode = "conf_aware"
        configs.append(
            {
                "label": DISPLAY_NAMES[key],
                "def_pct": float(row["adt_pct_deferred"]),
                "err_red": _err_reduction_pct(row),
                "runtime": float(row["avg_runtime_s"]),
                "color": METHOD_COLORS[method],
                "marker": MODE_MARKERS[mode],
            }
        )

    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    # Subtle grid for readability
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, color="#C9CED6", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    # Subtle shaded preferred operating region (no hard edges)
    ax.axvspan(0, 15, ymin=(55 - 15) / (100 - 15), ymax=1.0, alpha=0.06, color="#4DAA57", zorder=0)
    ax.text(
        7.5,
        96.5,
        "Preferred region",
        fontsize=8.2,
        ha="center",
        va="top",
        color="#1F6E3A",
        fontstyle="italic",
    )

    xs, ys, labels = [], [], []
    MARKER_SIZE = 210  # balanced against axis density
    for config in configs:
        ax.scatter(
            config["def_pct"],
            config["err_red"],
            s=MARKER_SIZE,
            c=config["color"],
            marker=config["marker"],
            edgecolors="#1F2937",
            linewidths=1.2,
            zorder=5,
            alpha=0.95,
        )
        xs.append(config["def_pct"])
        ys.append(config["err_red"])
        labels.append(config["label"])

    add_adjusted_labels(
        ax,
        xs,
        ys,
        labels,
        initial_offsets=[
            (1.4, -1.8),
            (1.6, 2.2),
            (1.4, -0.6),
            (1.2, 0.8),
            (1.3, 1.8),
            (-3.4, 1.4),
        ],
        text_kwargs={"fontsize": 7.8},
        arrow_kwargs={"arrowstyle": "->", "color": "#8A94A2", "lw": 0.7, "shrinkA": 8, "shrinkB": 5},
        expand_points=(2.1, 2.4),
        expand_text=(1.55, 1.80),
        force_points=0.80,
        force_text=0.90,
    )

    ax.set_xlabel("Deferral rate (% of pixels deferred)")
    ax.set_ylabel("Error reduction ratio (%)")
    ax.set_title("Error Reduction vs. Deferral Rate", pad=8)
    ax.set_xlim(0, 40)
    ax.set_ylim(15, 100)
    style_axes(ax, xpad=10, ypad=10)

    method_handles = [
        mpatches.Patch(color=METHOD_COLORS["mc_dropout"], label="MC Dropout"),
        mpatches.Patch(color=METHOD_COLORS["tta"], label="TTA"),
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#9AA4B2", markeredgecolor="#1F2937", label="Global", markersize=8),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#9AA4B2", markeredgecolor="#1F2937", label="Adaptive", markersize=8),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#9AA4B2", markeredgecolor="#1F2937", label="Confidence-aware", markersize=8),
    ]
    legend1 = ax.legend(
        handles=method_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.03),
        title="Uncertainty",
        frameon=True,
        fontsize=7.8,
        title_fontsize=8.2,
        framealpha=0.92,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=marker_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.22),
        title="Deferral policy",
        frameon=True,
        fontsize=7.8,
        title_fontsize=8.2,
        framealpha=0.92,
    )

    save_figure(fig, OUT_DIR / "summary_scatter.png")
    print("Generated: summary_scatter.png")


def make_error_reduction_bars() -> None:
    summary = _load_summary()
    order = [
        "mc_dropout",
        "mc_dropout_adaptive",
        "mc_dropout_conf_aware",
        "tta",
        "tta_adaptive",
        "tta_conf_aware",
    ]
    labels = [DISPLAY_NAMES[key] for key in order]
    err_before = [float(summary[key]["adt_error_before"]) for key in order]
    err_after = [float(summary[key]["adt_error_after"]) for key in order]
    err_pct = [_err_reduction_pct(summary[key]) for key in order]
    def_pct = [float(summary[key]["adt_pct_deferred"]) for key in order]
    colors_before = ["#B56C6C" if key.startswith("mc_") else "#C98557" for key in order]
    colors_after = ["#5F8BC4" if key.startswith("mc_") else "#6CBF8B" for key in order]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(12.3, 6.7), constrained_layout=True)

    before = ax.bar(
        x - width / 2,
        [e * 100 for e in err_before],
        width,
        label="Error before deferral",
        color=colors_before,
        edgecolor="#7B3640",
        linewidth=0.8,
    )
    after = ax.bar(
        x + width / 2,
        [e * 100 for e in err_after],
        width,
        label="Error after deferral",
        color=colors_after,
        edgecolor="#2F5D6D",
        linewidth=0.8,
    )

    label_xs = []
    label_ys = []
    label_texts = []
    for i, (erp, dp) in enumerate(zip(err_pct, def_pct)):
        ymax = max(before[i].get_height(), after[i].get_height())
        label_xs.append(float(i))
        label_ys.append(float(ymax + 0.18))
        label_texts.append(f"ERR {erp:.1f}%\nDef {dp:.1f}%")

    add_adjusted_labels(
        ax,
        label_xs,
        label_ys,
        label_texts,
        initial_offsets=[(0.0, 0.18), (0.0, 0.32), (0.0, 0.18), (0.0, 0.32), (0.0, 0.18), (0.0, 0.32)],
        text_kwargs={
            "fontsize": 8.0,
            "fontweight": "bold",
            "ha": "center",
            "va": "bottom",
            "bbox": dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#E2E8F0", alpha=0.94),
        },
        arrow_kwargs={"arrowstyle": "->", "color": "#7C8798", "lw": 0.75, "shrinkA": 4, "shrinkB": 3},
        expand_points=(1.45, 1.55),
        expand_text=(1.15, 1.28),
        force_points=0.4,
        force_text=0.5,
    )

    ax.set_ylabel("Pixel error rate (%)")
    ax.set_title("Error Rates Before and After Deferral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, max([e * 100 for e in err_before]) + 2.6)
    ax.margins(x=0.04)
    style_axes(ax)
    ax.legend(loc="upper right")

    save_figure(fig, OUT_DIR / "error_reduction_bars.png")
    print("Generated: error_reduction_bars.png")


def copy_existing_assets() -> None:
    copies = {
        COMPARISON_DIR / "risk_coverage_comparison.png": OUT_DIR / "risk_coverage.png",
        COMPARISON_DIR / "method_comparison_bars.png": OUT_DIR / "method_comparison_bars.png",
        COMPARISON_DIR / "qualitative_comparison.png": OUT_DIR / "qualitative_comparison.png",
        COMPARISON_DIR / "deferral_3mode_tta.png": OUT_DIR / "deferral_3mode_tta.png",
        COMPARISON_DIR / "deferral_3mode_mc.png": OUT_DIR / "deferral_3mode_mc.png",
    }
    for src, dst in copies.items():
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {dst.name}")


if __name__ == "__main__":
    make_pipeline()
    make_summary_scatter()
    make_error_reduction_bars()
    copy_existing_assets()
    print(f"\nAll figures saved to: {OUT_DIR}")
    print("Files:", sorted(p.name for p in OUT_DIR.iterdir()))
