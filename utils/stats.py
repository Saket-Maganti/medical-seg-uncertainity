"""
Statistical significance testing for segmentation + uncertainty metrics.

Implements:
  - Bootstrap confidence intervals (n=1000) for any scalar metric
  - Wilcoxon signed-rank test for paired comparison (MC Dropout vs Ensemble)
  - McNemar's test for comparing binary error patterns
  - Summary table with significance stars

Usage:
    from utils.stats import bootstrap_ci, wilcoxon_comparison, significance_table
"""

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from typing import Callable, List, Tuple
import json


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray,
                 metric_fn: Callable = np.mean,
                 n_bootstrap: int = 1000,
                 ci: float = 0.95,
                 seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric computed on a list of per-image values.

    Args:
        values:     Per-image metric values (e.g. Dice scores for 20 test images)
        metric_fn:  Aggregation function (default: mean)
        n_bootstrap: Number of bootstrap samples
        ci:         Confidence level (default 0.95)

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n   = len(values)
    estimates = [metric_fn(rng.choice(values, size=n, replace=True))
                 for _ in range(n_bootstrap)]

    alpha = 1.0 - ci
    lo = np.percentile(estimates, 100 * alpha / 2)
    hi = np.percentile(estimates, 100 * (1 - alpha / 2))
    return float(metric_fn(values)), float(lo), float(hi)


def bootstrap_ci_dict(per_image_metrics: List[dict],
                      keys: List[str],
                      n_bootstrap: int = 1000) -> dict:
    """
    Compute bootstrap CIs for multiple metrics at once.

    Args:
        per_image_metrics: List of dicts, one per test image
        keys: Metric names to compute CIs for

    Returns:
        dict: {metric: {"mean": x, "ci_lo": x, "ci_hi": x}}
    """
    results = {}
    for key in keys:
        vals = np.array([m[key] for m in per_image_metrics
                         if key in m and not np.isnan(m[key])])
        if len(vals) == 0:
            continue
        mean, lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap)
        results[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                        "ci_str": f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"}
    return results


# ── Wilcoxon signed-rank ──────────────────────────────────────────────────────

def wilcoxon_comparison(metrics_a: List[dict],
                         metrics_b: List[dict],
                         keys: List[str],
                         name_a: str = "MC Dropout",
                         name_b: str = "Deep Ensemble") -> dict:
    """
    Wilcoxon signed-rank test comparing two methods on paired per-image metrics.
    Appropriate because:
      - Same test images used for both methods (paired)
      - Non-parametric (no normality assumption)
      - DRIVE n=20 is too small for t-test

    Returns dict with statistic, p-value, and significance for each metric.
    """
    results = {}
    print(f"\n[Stats] Wilcoxon signed-rank: {name_a} vs {name_b}")
    print(f"  {'Metric':<20} {'Mean A':>8} {'Mean B':>8} "
          f"{'W-stat':>8} {'p-value':>10} {'Sig':>5}")
    print("  " + "-" * 65)

    for key in keys:
        a = np.array([m[key] for m in metrics_a if key in m])
        b = np.array([m[key] for m in metrics_b if key in m])

        if len(a) < 5 or len(b) < 5:
            continue

        # Wilcoxon requires no ties (add tiny jitter if needed)
        diff = a - b
        if np.all(diff == 0):
            continue

        try:
            stat, pval = wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            stat, pval = float("nan"), float("nan")

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 \
              else "*" if pval < 0.05 else "ns"

        results[key] = {
            "mean_a":    float(np.mean(a)),
            "mean_b":    float(np.mean(b)),
            "statistic": float(stat),
            "pval":      float(pval),
            "sig":       sig,
            "winner":    name_a if np.mean(a) > np.mean(b) else name_b,
        }

        print(f"  {key:<20} {np.mean(a):>8.4f} {np.mean(b):>8.4f} "
              f"{stat:>8.1f} {pval:>10.4f} {sig:>5}")

    return results


# ── McNemar's test ────────────────────────────────────────────────────────────

def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray,
                  labels: np.ndarray) -> dict:
    """
    McNemar's test: are the error patterns of two methods significantly different?
    Works on binary predictions (already thresholded).

    Args:
        preds_a, preds_b: Binary predictions (N,) — 0/1
        labels:           Ground truth (N,)

    Returns:
        dict with chi2 statistic, p-value, and interpretation
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # Contingency table
    b = (~correct_a &  correct_b).sum()  # A wrong, B correct
    c = ( correct_a & ~correct_b).sum()  # A correct, B wrong

    # McNemar with continuity correction
    if b + c == 0:
        return {"chi2": 0.0, "pval": 1.0, "sig": "ns",
                "b": int(b), "c": int(c)}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    pval = stats.chi2.sf(chi2, df=1)
    sig  = "***" if pval < 0.001 else "**" if pval < 0.01 \
           else "*" if pval < 0.05 else "ns"

    return {"chi2": float(chi2), "pval": float(pval), "sig": sig,
            "b": int(b), "c": int(c)}


# ── Summary table ─────────────────────────────────────────────────────────────

def significance_table(method_results: dict,
                        keys: List[str] = None,
                        n_bootstrap: int = 1000) -> str:
    """
    Print a publication-ready table:
    Method | Dice (95% CI) | AUC (95% CI) | ECE (95% CI) | Unc-AUROC (95% CI)

    Args:
        method_results: {method_name: [per_image_metric_dicts]}
        keys: Metric keys to include
    """
    if keys is None:
        keys = ["dice", "auc", "ece", "unc_auroc"]

    print("\n" + "=" * 80)
    print("RESULTS TABLE WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 80)
    header = f"{'Method':<22}" + "".join(f"{k:>18}" for k in keys)
    print(header)
    print("-" * 80)

    all_cis = {}
    for method_name, per_img in method_results.items():
        cis = bootstrap_ci_dict(per_img, keys, n_bootstrap=n_bootstrap)
        all_cis[method_name] = cis
        row = f"{method_name:<22}"
        for k in keys:
            if k in cis:
                c = cis[k]
                row += f"  {c['mean']:.3f} [{c['ci_lo']:.3f},{c['ci_hi']:.3f}]"
            else:
                row += f"  {'—':>16}"
        print(row)

    print("=" * 80)
    print("Format: mean [95% CI lower, upper]  |  Bootstrap n=1000")
    return all_cis


def save_stats_report(output: dict, path: str):
    """Save all statistical results to JSON."""
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[Stats] Saved to {path}")
