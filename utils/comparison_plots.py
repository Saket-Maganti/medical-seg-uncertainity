"""
Cross-method and cross-mode comparison visualizations.

Functions
---------
plot_risk_coverage_comparison
    Overlay risk-coverage curves for MC Dropout vs TTA (and any other methods).

plot_deferral_mode_comparison
    Overlay global vs image-adaptive deferral curves for a single method.

plot_qualitative_comparison
    4-panel qualitative example: image, prediction, uncertainty, deferred regions.
    Accepts results from two methods side-by-side.

plot_method_summary_bars
    Bar chart comparing aggregate metrics across methods.

All functions write .png files to `output_dir` and return the save path.
"""

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from utils.figure_style import add_adjusted_labels, apply_publication_style, save_figure, style_axes


apply_publication_style()

# ── shared style ──────────────────────────────────────────────────────────────

_METHOD_COLORS = {
    "mc_dropout":  "#4C72B0",
    "tta":         "#DD8452",
    "deterministic": "#8C8C8C",
    "ensemble":    "#55A868",
}
_MODE_COLORS = {
    "global":   "#4C72B0",
    "adaptive": "#E07B54",
}
_MODE_MARKERS = {
    "global": "o",
    "adaptive": "^",
    "conf_aware": "s",
}

_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm(image_chw: np.ndarray) -> np.ndarray:
    return (image_chw.transpose(1, 2, 0) * _IMG_STD + _IMG_MEAN).clip(0.0, 1.0)


def _load_risk_coverage_csv(path: str) -> Optional[dict]:
    """Load a risk_coverage.csv into dict of lists."""
    p = Path(path)
    if not p.exists():
        return None
    rows = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None
    return {
        "coverage":   [float(r["coverage"])   for r in rows],
        "error_rate": [float(r["error_rate"]) for r in rows],
        "dice":       [float(r["dice"])        for r in rows],
    }


# ── 1. Risk-coverage comparison (MC Dropout vs TTA) ───────────────────────────

def plot_risk_coverage_comparison(
    methods: Dict[str, str],   # {method_label: path_to_risk_coverage.csv}
    output_dir: str,
    title: str = "Risk-Coverage Comparison",
) -> str:
    """
    Overlay risk-coverage and coverage-Dice curves for multiple methods.

    Args:
        methods: dict mapping display name to path of risk_coverage.csv
        output_dir: where to save the figure

    Returns:
        path to saved PNG
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plotted = 0

    for label, csv_path in methods.items():
        data = _load_risk_coverage_csv(csv_path)
        if data is None:
            print(f"  [comparison_plots] skipped {label}: {csv_path} not found")
            continue
        key = label.lower().replace(" ", "_")
        color = _METHOD_COLORS.get(key, f"C{plotted}")

        axes[0].plot(data["coverage"], data["error_rate"],
                     color=color, linewidth=2.5, label=label)
        axes[1].plot(data["coverage"], data["dice"],
                     color=color, linewidth=2.5, label=label)
        plotted += 1

    # Risk-coverage panel
    axes[0].set_xlabel("Coverage (fraction accepted)", fontsize=12)
    axes[0].set_ylabel("Risk (pixel error rate)", fontsize=12)
    axes[0].set_title("Risk-Coverage\n(lower risk at lower coverage = uncertainty works)",
                       fontsize=11)
    axes[0].invert_xaxis()          # leftward = more deferral
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(1.0, 0.1)

    # Coverage-Dice panel
    axes[1].axhline(0.82, color="red", linestyle="--", linewidth=1.2,
                    label="Dice = 0.82 target")
    axes[1].set_xlabel("Coverage (fraction accepted)", fontsize=12)
    axes[1].set_ylabel("Dice on accepted pixels", fontsize=12)
    axes[1].set_title("Coverage vs Dice", fontsize=11)
    axes[1].invert_xaxis()
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(1.0, 0.1)

    plt.suptitle(title, fontsize=13)
    style_axes(axes[0])
    style_axes(axes[1])
    save_path = output_dir / "risk_coverage_comparison.png"
    save_figure(fig, save_path)
    print(f"  Saved: {save_path}")
    return str(save_path)


# ── 2. Global vs adaptive deferral comparison ─────────────────────────────────

def plot_deferral_mode_comparison(
    global_csv: str,
    adaptive_csv: str,
    output_dir: str,
    method_name: str = "MC Dropout",
) -> str:
    """
    Overlay risk-coverage and coverage-Dice curves for global vs
    image-adaptive deferral on the same method.

    Args:
        global_csv:   path to risk_coverage.csv from global deferral
        adaptive_csv: path to risk_coverage.csv from image-adaptive deferral
        output_dir:   where to save the figure
        method_name:  display name of the method

    Returns:
        path to saved PNG
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_data   = _load_risk_coverage_csv(global_csv)
    adaptive_data = _load_risk_coverage_csv(adaptive_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for data, label, color in [
        (global_data,   "Global threshold",         _MODE_COLORS["global"]),
        (adaptive_data, "Image-adaptive threshold", _MODE_COLORS["adaptive"]),
    ]:
        if data is None:
            continue
        axes[0].plot(data["coverage"], data["error_rate"],
                     color=color, linewidth=2.5, label=label)
        axes[1].plot(data["coverage"], data["dice"],
                     color=color, linewidth=2.5, label=label)

    for ax in axes:
        ax.invert_xaxis()
        ax.legend(fontsize=10)
        ax.set_xlim(1.0, 0.1)
        ax.set_xlabel("Coverage (fraction accepted)", fontsize=12)
        style_axes(ax)

    axes[0].set_ylabel("Risk (pixel error rate)", fontsize=12)
    axes[0].set_title("Risk-Coverage\nGlobal vs Image-Adaptive Deferral", fontsize=11)

    axes[1].axhline(0.82, color="red", linestyle="--", linewidth=1.2,
                    label="Dice = 0.82 target")
    axes[1].legend(fontsize=10)
    axes[1].set_ylabel("Dice on accepted pixels", fontsize=12)
    axes[1].set_title("Coverage vs Dice\nGlobal vs Image-Adaptive Deferral", fontsize=11)

    plt.suptitle(f"{method_name}: Deferral Mode Comparison", fontsize=13)
    save_path = output_dir / "deferral_mode_comparison.png"
    save_figure(fig, save_path)
    print(f"  Saved: {save_path}")
    return str(save_path)


# ── 3. Qualitative comparison (one image, two methods) ────────────────────────

def plot_qualitative_comparison(
    image_chw: np.ndarray,
    gt_mask: np.ndarray,
    pred_mc: np.ndarray,
    unc_mc: np.ndarray,
    deferred_mc: np.ndarray,
    pred_tta: np.ndarray,
    unc_tta: np.ndarray,
    deferred_tta: np.ndarray,
    save_path: str,
    threshold: float = 0.5,
) -> str:
    """
    Two-row, five-column qualitative comparison figure.

    Row 0 — MC Dropout:  image | GT | prediction | uncertainty | deferred regions
    Row 1 — TTA:         image | GT | prediction | uncertainty | deferred regions

    Args:
        image_chw:    (C, H, W) normalised image array
        gt_mask:      (H, W) binary ground truth
        pred_mc/tta:  (H, W) predicted probabilities
        unc_mc/tta:   (H, W) uncertainty maps
        deferred_mc/tta: (H, W) binary deferred masks (1 = deferred)
        save_path:    output PNG path
        threshold:    binarisation threshold for predictions

    Returns:
        path to saved PNG
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img_rgb = _denorm(image_chw)

    def _overlay(img, pred_bin, deferred):
        H, W = pred_bin.shape
        ov = np.zeros((H, W, 3), dtype=np.float32)
        ov[..., 1] = pred_bin * (1.0 - deferred)   # green = accepted
        ov[..., 0] = deferred                        # red   = deferred
        canvas = img.copy()
        alpha = 0.55
        canvas = canvas * (1 - alpha * (ov.sum(-1, keepdims=True) > 0)) + ov * alpha
        return canvas.clip(0, 1)

    def _conf(unc):
        c = 1.0 - unc
        lo, hi = c.min(), c.max()
        return (c - lo) / (hi - lo + 1e-8)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    titles = ["Input", "Ground truth", "Prediction", "Uncertainty", "Deferred regions"]

    for row, (method, pred, unc, deferred) in enumerate([
        ("MC Dropout", pred_mc, unc_mc, deferred_mc),
        ("TTA",        pred_tta, unc_tta, deferred_tta),
    ]):
        pred_bin = (pred > threshold).astype(np.float32)

        axes[row, 0].imshow(img_rgb)
        axes[row, 1].imshow(gt_mask, cmap="gray")
        axes[row, 2].imshow(pred_bin, cmap="gray")
        axes[row, 2].contour(gt_mask, levels=[0.5], colors="lime", linewidths=0.8)

        im = axes[row, 3].imshow(unc, cmap="hot")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

        axes[row, 4].imshow(_overlay(img_rgb, pred_bin, deferred))
        pct = deferred.mean() * 100.0
        accepted_p = mpatches.Patch(color="green", alpha=0.7, label="Accepted")
        deferred_p = mpatches.Patch(color="red",   alpha=0.7,
                                     label=f"Deferred ({pct:.1f}%)")
        axes[row, 4].legend(handles=[accepted_p, deferred_p],
                             loc="lower right", fontsize=8)

        for col, title in enumerate(titles):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=11)

        axes[row, 0].set_ylabel(method, fontsize=12, rotation=90, labelpad=8)

    plt.suptitle("Qualitative Comparison: MC Dropout vs TTA", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    return str(save_path)


# ── 4. Bar chart: metric comparison across methods ────────────────────────────

def plot_method_summary_bars(
    results: Dict[str, Dict],   # {method_label: summary_dict}
    metrics: List[str],
    output_dir: str,
    title: str = "Method Comparison",
) -> str:
    """
    Multi-panel bar chart comparing scalar metrics across methods.

    Args:
        results: dict of {label: summary_dict} where summary_dict has
                 the requested metric keys.
        metrics: list of metric keys to plot (e.g. ["dice", "global_ece", ...])
        output_dir: output directory

    Returns:
        path to saved PNG
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_metrics = len(metrics)
    ncols = min(n_metrics, 4)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.9 * ncols, 4.4 * nrows),
        constrained_layout=True,
    )
    if n_metrics == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    labels  = list(results.keys())
    x       = np.arange(len(labels))
    colors  = [_METHOD_COLORS.get(lbl.lower().replace(" ", "_"), f"C{i}")
               for i, lbl in enumerate(labels)]
    pretty_labels = [
        lbl.replace("mc_dropout", "MC")
        .replace("tta", "TTA")
        .replace("_adaptive", "\nAdaptive")
        .replace("_conf_aware", "\nConf-aware")
        .replace("_calibrated", "\nTemp-scaled")
        .replace("_", " ")
        for lbl in labels
    ]

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax   = axes[r, c]
        vals = [float(results[lbl].get(metric, float("nan"))) for lbl in labels]
        bars = ax.bar(x, vals, color=colors, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(pretty_labels, fontsize=9.5, rotation=15, ha="right")
        ax.set_title(metric.replace("_", " ").upper(), fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        style_axes(ax)

        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            lo = min(finite_vals)
            hi = max(finite_vals)
            span = max(hi - lo, 1e-6)
            if lo >= 0:
                ax.set_ylim(0, hi + span * 0.18 + 0.01 * max(1.0, hi))
            else:
                ax.set_ylim(lo - span * 0.12, hi + span * 0.18)

        # Annotate bars
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                y = bar.get_height()
                offset = max(abs(y) * 0.02, 0.002)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y + offset,
                    f"{v:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                )

    # Hide unused subplots
    for idx in range(n_metrics, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    plt.suptitle(title, fontsize=13)
    save_path = output_dir / "method_comparison_bars.png"
    save_figure(fig, save_path)
    print(f"  Saved: {save_path}")
    return str(save_path)


def plot_deferral_policy_operating_points(
    results: Dict[str, Dict],
    output_dir: str,
    method_name: str = "TTA",
) -> str:
    """
    Compare three policy operating points for a single uncertainty method.

    The left panel shows the local design space (deferral rate vs error reduction);
    the right panel shows deferral efficiency, i.e. relative error reduction per
    1% deferred. This avoids the misleading reuse of risk-coverage curves across
    policy families while still communicating the true policy differences.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_slug = method_name.lower().replace(" ", "_")
    order = [
        ("global", "Global"),
        ("adaptive", "Adaptive"),
        ("conf_aware", "Confidence-aware"),
    ]

    selected = []
    for key, display in order:
        summary = results.get(f"{method_slug}_{key}" if key != "global" else method_slug)
        if summary is None:
            continue
        err_before = float(summary["adt_error_before"])
        err_after = float(summary["adt_error_after"])
        err_reduction_pct = 100.0 * (err_before - err_after) / max(err_before, 1e-12)
        def_pct = float(summary["adt_pct_deferred"])
        efficiency = err_reduction_pct / max(def_pct, 1e-9)
        selected.append(
            {
                "key": key,
                "label": display,
                "def_pct": def_pct,
                "err_reduction_pct": err_reduction_pct,
                "err_before": err_before,
                "err_after": err_after,
                "efficiency": efficiency,
            }
        )

    if not selected:
        raise ValueError(f"No operating-point summaries found for {method_name!r}.")

    method_color = _METHOD_COLORS.get(method_slug, "#4C72B0")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=True)

    # Left: operating-point design space.
    ax = axes[0]
    def_vals = [row["def_pct"] for row in selected]
    red_vals = [row["err_reduction_pct"] for row in selected]
    max_x = max(def_vals) + 6.0
    min_y = max(0.0, min(red_vals) - 8.0)
    max_y = min(100.0, max(red_vals) + 10.0)

    ax.axvspan(0, min(max_x, 15.0), color="#4DAA57", alpha=0.08, zorder=0)
    ax.axhspan(max(0.0, max_y - 20.0), max_y, color="#4DAA57", alpha=0.05, zorder=0)

    xs, ys, labels = [], [], []
    for row in selected:
        marker = _MODE_MARKERS[row["key"]]
        ax.scatter(
            row["def_pct"],
            row["err_reduction_pct"],
            s=240,
            marker=marker,
            c=method_color,
            edgecolors="white",
            linewidths=1.6,
            zorder=4,
            alpha=0.95,
        )
        ax.scatter(
            row["def_pct"],
            row["err_reduction_pct"],
            s=240,
            marker=marker,
            facecolors="none",
            edgecolors="#1F2937",
            linewidths=0.9,
            zorder=5,
        )
        xs.append(row["def_pct"])
        ys.append(row["err_reduction_pct"])
        labels.append(row["label"])

    initial_offsets = [(-0.8, 2.5), (0.8, 2.0), (0.8, -1.4)]
    add_adjusted_labels(
        ax,
        xs,
        ys,
        labels,
        initial_offsets=initial_offsets[: len(labels)],
        text_kwargs={"fontsize": 8.6},
        arrow_kwargs={"arrowstyle": "->", "color": "#6B7280", "lw": 0.8},
        expand_points=(1.6, 1.8),
        expand_text=(1.3, 1.5),
        force_points=0.55,
        force_text=0.65,
    )

    ax.text(
        0.03,
        0.97,
        "Preferred corner:\nhigher reduction,\nlower deferral",
        transform=ax.transAxes,
        va="top",
        fontsize=8.8,
        color="#2E8B57",
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#D8E8DC", alpha=0.94),
    )
    ax.set_xlabel("Deferral rate (% of pixels deferred)")
    ax.set_ylabel("Error reduction ratio (%)")
    ax.set_title(f"{method_name}: policy operating points")
    ax.set_xlim(0, max_x)
    ax.set_ylim(min_y, max_y)
    style_axes(ax, xpad=10, ypad=10)

    # Right: efficiency + after-deferral error.
    ax = axes[1]
    y = np.arange(len(selected))
    efficiencies = [row["efficiency"] for row in selected]
    err_after = [100.0 * row["err_after"] for row in selected]
    base_error = 100.0 * selected[0]["err_before"]

    bars = ax.barh(y, efficiencies, color=method_color, alpha=0.88, edgecolor="#1F2937")
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in selected])
    ax.invert_yaxis()
    ax.set_xlabel("Efficiency (reduction % per 1% deferred)")
    ax.set_title("Review efficiency")
    style_axes(ax, xpad=10, ypad=10)

    err_ax = ax.twiny()
    err_ax.plot(err_after, y, color="#C44E52", marker="o", linewidth=2.0, label="Error after deferral")
    err_ax.axvline(base_error, color="#7C8798", linestyle="--", linewidth=1.2, label="Error before deferral")
    err_ax.set_xlabel("Pixel error rate after deferral (%)")
    err_ax.grid(False)

    for bar, row in zip(bars, selected):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{row['efficiency']:.2f}x",
            va="center",
            fontsize=8.5,
        )
        err_ax.text(
            row["err_after"] * 100.0 + 0.05,
            bar.get_y() + bar.get_height() / 2 - 0.1,
            f"{row['err_after'] * 100.0:.2f}%",
            color="#8E2F33",
            fontsize=8.2,
            va="center",
        )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=method_color, markeredgecolor="#1F2937", label=method_name, markersize=8),
        plt.Line2D([0], [0], color="#C44E52", marker="o", label="Error after deferral"),
        plt.Line2D([0], [0], color="#7C8798", linestyle="--", label="Error before deferral"),
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8.8)

    filename_slug = "mc" if method_slug == "mc_dropout" else method_slug
    save_path = output_dir / f"deferral_3mode_{filename_slug}.png"
    save_figure(fig, save_path)
    print(f"  Saved: {save_path}")
    return str(save_path)


# ── 5. Reliability diagram: before vs after calibration ───────────────────────

def plot_calibration_comparison(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    temperature: float = None,
    method_name: str = "",
) -> str:
    """
    Two-panel reliability diagram comparing uncalibrated vs temperature-scaled probs.

    Args:
        probs_before: (N,) raw probabilities
        probs_after:  (N,) calibrated probabilities
        labels:       (N,) binary ground-truth labels
        output_dir:   where to save the figure
        temperature:  learned T (used in title)
        method_name:  e.g. "MC Dropout" or "TTA"

    Returns:
        path to saved PNG
    """
    from utils.metrics import expected_calibration_error, reliability_diagram_data

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ece_before = expected_calibration_error(probs_before, labels)
    ece_after  = expected_calibration_error(probs_after,  labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, probs, ece, title in [
        (axes[0], probs_before, ece_before,
         f"Before temperature scaling\nECE = {ece_before:.4f}"),
        (axes[1], probs_after,  ece_after,
         f"After temperature scaling\nECE = {ece_after:.4f}"),
    ]:
        centers, conf, acc, _ = reliability_diagram_data(probs, labels)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
        ax.bar(centers, acc, width=1.0 / len(centers), alpha=0.65,
               color="#4C72B0", label="Accuracy")
        ax.plot(centers, conf, "ro-", markersize=4, label="Confidence")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        style_axes(ax)

    t_str = f" (T = {temperature:.3f})" if temperature is not None else ""
    plt.suptitle(f"{method_name} Calibration Comparison{t_str}", fontsize=13)
    save_path = output_dir / "calibration_comparison.png"
    save_figure(fig, save_path)
    print(f"  Saved: {save_path}")
    return str(save_path)


# ── 6. 3-mode deferral risk-coverage overlay ──────────────────────────────────

def plot_deferral_3mode_comparison(
    global_csv: str,
    adaptive_csv: str,
    conf_aware_csv: str,
    output_dir: str,
    method_name: str = "MC Dropout",
) -> str:
    """
    Overlay risk-coverage curves for three deferral modes:
    global threshold, image-adaptive, and confidence-aware.

    Args:
        global_csv:     path to risk_coverage.csv from global deferral
        adaptive_csv:   path to risk_coverage.csv from image-adaptive deferral
        conf_aware_csv: path to risk_coverage.csv from confidence-aware deferral
        output_dir:     where to save the figure
        method_name:    display name of the method

    Returns:
        path to saved PNG
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _MODE_3_COLORS = {
        "global":     "#4C72B0",
        "adaptive":   "#E07B54",
        "conf_aware": "#55A868",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for csv_path, label, color in [
        (global_csv,     "Global threshold",   _MODE_3_COLORS["global"]),
        (adaptive_csv,   "Image-adaptive",      _MODE_3_COLORS["adaptive"]),
        (conf_aware_csv, "Confidence-aware",    _MODE_3_COLORS["conf_aware"]),
    ]:
        data = _load_risk_coverage_csv(csv_path)
        if data is None:
            print(f"  [comparison_plots] skipped {label}: {csv_path} not found")
            continue
        axes[0].plot(data["coverage"], data["error_rate"],
                     color=color, linewidth=2.5, label=label)
        axes[1].plot(data["coverage"], data["dice"],
                     color=color, linewidth=2.5, label=label)

    for ax in axes:
        ax.invert_xaxis()
        ax.legend(fontsize=10)
        ax.set_xlim(1.0, 0.1)
        ax.set_xlabel("Coverage (fraction accepted)", fontsize=12)
        style_axes(ax)

    axes[0].set_ylabel("Risk (pixel error rate)", fontsize=12)
    axes[0].set_title("Risk-Coverage: 3 Deferral Modes", fontsize=11)

    axes[1].axhline(0.82, color="red", linestyle="--", linewidth=1.2,
                    label="Dice = 0.82 target")
    axes[1].legend(fontsize=10)
    axes[1].set_ylabel("Dice on accepted pixels", fontsize=12)
    axes[1].set_title("Coverage vs Dice: 3 Deferral Modes", fontsize=11)

    plt.suptitle(f"{method_name}: Deferral Mode Comparison (3 modes)", fontsize=13)
    save_path = output_dir / "deferral_3mode_comparison.png"
    save_figure(fig, save_path)
    print(f"  Saved: {save_path}")
    return str(save_path)
