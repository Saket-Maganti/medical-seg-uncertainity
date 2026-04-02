"""
Deferral Policy for Uncertainty-Guided Human-in-the-Loop Review.

Core idea:
  If pixel uncertainty > threshold T → flag for human review (defer)
  Otherwise → accept model prediction automatically

This converts uncertainty maps into a clinically actionable tool:
  - Clinician chooses operating point on the PR curve
  - High precision: only flag when truly uncertain (fewer reviews, more misses)
  - High recall: flag everything uncertain (more reviews, fewer misses)

Outputs:
  - Precision-Recall curve across all T values
  - Coverage vs Dice tradeoff curve
  - Optimal threshold by F1, by precision@recall, by coverage budget
  - JSON summary of deferral statistics

Usage:
    from utils.deferral import DeferralPolicy
    policy = DeferralPolicy(uncertainty_maps, error_masks, predictions, gt_masks)
    policy.run(output_dir="results/deferral")
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from pathlib import Path
import json


class DeferralPolicy:
    """
    Builds and evaluates a pixel-level deferral policy.

    Args:
        uncertainty_maps: list of (H,W) arrays — epistemic uncertainty per image
        error_masks:      list of (H,W) binary arrays — 1 where model is wrong
        pred_probs:       list of (H,W) arrays — model predicted probabilities
        gt_masks:         list of (H,W) binary arrays — ground truth
        fov_masks:        list of (H,W) binary arrays — field of view (optional)
    """

    def __init__(self, uncertainty_maps, error_masks, pred_probs,
                 gt_masks, fov_masks=None):
        self.uncertainty_maps = uncertainty_maps
        self.error_masks      = error_masks
        self.pred_probs       = pred_probs
        self.gt_masks         = gt_masks
        self.fov_masks        = fov_masks or [None] * len(uncertainty_maps)

        # Flatten all pixels (within FOV)
        self._flatten()

    def _flatten(self):
        unc_all, err_all, pred_all, gt_all = [], [], [], []
        for i in range(len(self.uncertainty_maps)):
            fov = self.fov_masks[i]
            u = self.uncertainty_maps[i]
            e = self.error_masks[i]
            p = self.pred_probs[i]
            g = self.gt_masks[i]
            if fov is not None:
                mask = fov.astype(bool)
                u, e, p, g = u[mask], e[mask], p[mask], g[mask]
            unc_all.append(u.ravel())
            err_all.append(e.ravel())
            pred_all.append(p.ravel())
            gt_all.append(g.ravel())

        self.unc_flat  = np.concatenate(unc_all)
        self.err_flat  = np.concatenate(err_all).astype(np.float32)
        self.pred_flat = np.concatenate(pred_all)
        self.gt_flat   = np.concatenate(gt_all)
        self.n_pixels  = len(self.unc_flat)

    # ── Core sweep ────────────────────────────────────────────────────────────

    def sweep_thresholds(self, n_thresholds: int = 200):
        """
        Sweep uncertainty threshold T from 0 to max.
        For each T, compute:
          - coverage:   fraction of pixels NOT deferred (accepted automatically)
          - precision:  of deferred pixels, fraction that are actually errors
          - recall:     of all error pixels, fraction that were deferred
          - dice_accepted: Dice on accepted (non-deferred) pixels only
        """
        unc = self.unc_flat[np.isfinite(self.unc_flat)]
        if unc.size == 0:
            raise ValueError("Uncertainty array is empty or non-finite; cannot sweep deferral thresholds.")

        # Use percentile-based thresholds so the sweep remains meaningful even
        # when uncertainty values are tightly clustered near zero.
        percentiles = np.linspace(1, 99, n_thresholds)
        thresholds = np.unique(np.percentile(unc, percentiles))
        if thresholds.size == 0:
            thresholds = np.array([float(np.mean(unc))], dtype=np.float32)
        results = []

        for T in thresholds:
            deferred  = self.unc_flat > T           # flagged for human review
            accepted  = ~deferred                   # auto-accepted

            coverage  = accepted.mean()             # fraction auto-accepted
            n_deferred = deferred.sum()

            if n_deferred == 0:
                precision = 1.0
            else:
                precision = self.err_flat[deferred].mean()  # PPV of deferral

            total_errors = self.err_flat.sum()
            if total_errors == 0:
                recall = 1.0
            else:
                recall = self.err_flat[deferred].sum() / total_errors

            # Dice on accepted pixels only
            if accepted.sum() > 0:
                p_acc = (self.pred_flat[accepted] > 0.5).astype(np.float32)
                g_acc = self.gt_flat[accepted]
                inter = (p_acc * g_acc).sum()
                dice_acc = (2 * inter + 1e-6) / (p_acc.sum() + g_acc.sum() + 1e-6)
            else:
                dice_acc = float("nan")

            f1 = (2 * precision * recall / (precision + recall + 1e-9))

            results.append({
                "threshold":      float(T),
                "coverage":       float(coverage),
                "precision":      float(precision),
                "recall":         float(recall),
                "f1":             float(f1),
                "dice_accepted":  float(dice_acc),
                "n_deferred":     int(n_deferred),
                "pct_deferred":   float(n_deferred / self.n_pixels * 100),
            })

        self.sweep_results = results
        return results

    def find_optimal_thresholds(self):
        """Find T that optimises different clinical objectives."""
        rs = self.sweep_results

        # Best F1 (balanced precision-recall)
        best_f1     = max(rs, key=lambda x: x["f1"])

        # Best precision while recall >= 0.8 (catch 80% of errors with fewest flags)
        hi_recall   = [r for r in rs if r["recall"] >= 0.8]
        best_prec   = max(hi_recall, key=lambda x: x["precision"]) if hi_recall else best_f1

        # Best coverage while dice_accepted >= 0.82
        hi_dice     = [r for r in rs if r["dice_accepted"] >= 0.82]
        best_cov    = max(hi_dice, key=lambda x: x["coverage"]) if hi_dice else best_f1

        self.optimal = {
            "max_f1":           best_f1,
            "precision_at_80recall": best_prec,
            "max_coverage_at_dice_82": best_cov,
            "high_precision": max(rs, key=lambda x: (x["precision"], x["recall"])),
            "high_recall": max(rs, key=lambda x: (x["recall"], x["precision"])),
        }
        return self.optimal

    # ── Plots ─────────────────────────────────────────────────────────────────

    def plot_all(self, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._plot_pr_curve(output_dir / "deferral_pr_curve.png")
        self._plot_coverage_dice(output_dir / "deferral_coverage_dice.png")
        self._plot_deferral_summary(output_dir / "deferral_summary.png")

    def _plot_pr_curve(self, save_path):
        rs = self.sweep_results
        precision = [r["precision"] for r in rs]
        recall    = [r["recall"]    for r in rs]
        unique_points = list(dict.fromkeys(zip(recall, precision)))
        if len(unique_points) >= 2:
            recall_sorted, precision_sorted = zip(*sorted(unique_points))
            pr_auc = auc(recall_sorted, precision_sorted)
        else:
            pr_auc = float("nan")

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color="#4C72B0", linewidth=2,
                label=(
                    f"Deferral PR curve (AUC={pr_auc:.3f})"
                    if np.isfinite(pr_auc)
                    else "Deferral PR curve (AUC=N/A)"
                ))
        if len(recall) >= 2:
            ax.fill_between(recall, precision, alpha=0.12, color="#4C72B0")

        # Mark optimal points
        opt = self.optimal
        for name, r, color, marker in [
            ("Max F1",  opt["max_f1"],           "#E07B54", "o"),
            ("Prec@80R", opt["precision_at_80recall"], "#54A87A", "s"),
            ("MaxCov@Dice82", opt["max_coverage_at_dice_82"], "#9B59B6", "^"),
        ]:
            ax.plot(r["recall"], r["precision"], marker=marker, color=color,
                    markersize=10, label=f"{name}  T={r['threshold']:.4f}  "
                    f"cov={r['coverage']:.2f}")

        ax.set_xlabel("Recall (fraction of errors caught)", fontsize=12)
        ax.set_ylabel("Precision (fraction of flags that are errors)", fontsize=12)
        ax.set_title("Deferral Policy: Precision-Recall Curve\n"
                     "Clinician chooses operating point", fontsize=13)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    def _plot_coverage_dice(self, save_path):
        rs = self.sweep_results
        coverage  = [r["coverage"]      for r in rs]
        dice_acc  = [r["dice_accepted"] for r in rs]
        pct_def   = [r["pct_deferred"]  for r in rs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Coverage vs Dice
        ax1.plot(coverage, dice_acc, color="#4C72B0", linewidth=2)
        ax1.axhline(0.82, color="#E07B54", linestyle="--", linewidth=1.2,
                    label="Dice = 0.82 target")
        ax1.set_xlabel("Coverage (fraction auto-accepted)", fontsize=12)
        ax1.set_ylabel("Dice on accepted pixels", fontsize=12)
        ax1.set_title("Coverage vs Dice Tradeoff", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # % deferred vs recall
        recall = [r["recall"] for r in rs]
        ax2.plot(pct_def, recall, color="#54A87A", linewidth=2)
        ax2.set_xlabel("% pixels deferred to human", fontsize=12)
        ax2.set_ylabel("Recall (errors caught)", fontsize=12)
        ax2.set_title("Deferral Rate vs Error Recall", fontsize=13)
        ax2.grid(alpha=0.3)

        plt.suptitle("Clinical Deferral Tradeoffs", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    def _plot_deferral_summary(self, save_path):
        """3-panel summary: threshold vs precision, recall, coverage."""
        rs = self.sweep_results
        T   = [r["threshold"]  for r in rs]
        P   = [r["precision"]  for r in rs]
        R   = [r["recall"]     for r in rs]
        C   = [r["coverage"]   for r in rs]
        F1  = [r["f1"]         for r in rs]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, y, label, color in [
            (axes[0], P,  "Precision", "#4C72B0"),
            (axes[1], R,  "Recall",    "#54A87A"),
            (axes[2], C,  "Coverage",  "#9B59B6"),
        ]:
            ax.plot(T, y, color=color, linewidth=2)
            ax.set_xlabel("Uncertainty threshold T", fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f"{label} vs T", fontsize=12)
            ax.grid(alpha=0.3)

        plt.suptitle("Effect of Deferral Threshold on Clinical Metrics", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # ── Run all ───────────────────────────────────────────────────────────────

    def run(self, output_dir: str = "results/deferral"):
        print("\n[DeferralPolicy] Sweeping thresholds...")
        self.sweep_thresholds()
        print("[DeferralPolicy] Finding optimal thresholds...")
        self.find_optimal_thresholds()
        print("[DeferralPolicy] Generating plots...")
        self.plot_all(output_dir)

        summary = {
            "n_pixels":   self.n_pixels,
            "optimal":    self.optimal,
            "operating_points": {
                "balanced_f1": self.optimal["max_f1"],
                "high_precision": self.optimal["high_precision"],
                "high_recall": self.optimal["high_recall"],
            },
        }
        out = Path(output_dir) / "deferral_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n[DeferralPolicy] Optimal thresholds:")
        for name, r in self.optimal.items():
            print(f"  {name:35s}: T={r['threshold']:.5f} | "
                  f"prec={r['precision']:.3f} | rec={r['recall']:.3f} | "
                  f"cov={r['coverage']:.3f} | dice_acc={r['dice_accepted']:.3f}")
        return summary
