"""
Selective Prediction for Uncertainty-Guided Abstention.

Selective prediction = the model can "abstain" on samples (or pixels)
it is uncertain about, improving accuracy on the cases it does answer.

Key curves:
  - Coverage vs Dice:    As we abstain on more uncertain pixels, Dice improves
  - Coverage vs AUC:     Same for AUC
  - Risk-Coverage curve: Expected error rate vs fraction accepted

This formalizes the deferral policy with a theoretical framework
(Geifman & El-Yaniv, 2017 / Selective Prediction paper).

Usage:
    from utils.selective_prediction import SelectivePrediction
    sp = SelectivePrediction(pred_probs, gt_masks, uncertainty_maps)
    sp.run(output_dir="results/selective")
"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from utils.figure_style import apply_publication_style, save_figure, style_axes


apply_publication_style()
from utils.metrics import dice_coefficient, compute_auc


class SelectivePrediction:
    """
    Computes coverage-accuracy tradeoff curves for pixel-level selective prediction.

    At coverage c:
      - Accept (100*c)% of pixels with lowest uncertainty
      - Compute segmentation metrics on accepted pixels only
    """

    def __init__(self, pred_probs, gt_masks, uncertainty_maps, fov_masks=None):
        """
        Args:
            pred_probs:       list of (H,W) predicted probabilities
            gt_masks:         list of (H,W) binary ground truth
            uncertainty_maps: list of (H,W) uncertainty values
            fov_masks:        list of (H,W) optional FOV masks
        """
        self.pred_probs       = pred_probs
        self.gt_masks         = gt_masks
        self.uncertainty_maps = uncertainty_maps
        self.fov_masks        = fov_masks or [None] * len(pred_probs)

        self._flatten()

    def _flatten(self):
        pred_all, gt_all, unc_all = [], [], []
        for i in range(len(self.pred_probs)):
            fov = self.fov_masks[i]
            p = self.pred_probs[i]
            g = self.gt_masks[i]
            u = self.uncertainty_maps[i]
            if fov is not None:
                mask = fov.astype(bool)
                p, g, u = p[mask], g[mask], u[mask]
            pred_all.append(p.ravel())
            gt_all.append(g.ravel())
            unc_all.append(u.ravel())

        self.pred_flat = np.concatenate(pred_all)
        self.gt_flat   = np.concatenate(gt_all).astype(np.float32)
        self.unc_flat  = np.concatenate(unc_all)

        # Sort by uncertainty (ascending = most certain first)
        self.sort_idx  = np.argsort(self.unc_flat)

    def sweep_coverage(self, n_points: int = 100) -> list:
        """
        Sweep coverage from 10% to 100%.
        For each coverage level, compute metrics on accepted pixels.
        """
        coverages = np.linspace(0.1, 1.0, n_points)
        n_total   = len(self.pred_flat)
        results   = []

        for cov in coverages:
            n_accept  = max(1, int(cov * n_total))
            accept_idx = self.sort_idx[:n_accept]

            p_acc = self.pred_flat[accept_idx]
            g_acc = self.gt_flat[accept_idx]

            # Dice on accepted
            p_bin = (p_acc > 0.5).astype(np.float32)
            inter = (p_bin * g_acc).sum()
            dice  = (2 * inter + 1e-6) / (p_bin.sum() + g_acc.sum() + 1e-6)

            # AUC on accepted (need some pos+neg)
            if g_acc.sum() > 0 and g_acc.sum() < len(g_acc):
                from sklearn.metrics import roc_auc_score
                try:
                    auc_val = roc_auc_score(g_acc, p_acc)
                except:
                    auc_val = float("nan")
            else:
                auc_val = float("nan")

            # Error rate (risk)
            error_rate = float((p_bin != g_acc).mean())

            # Max uncertainty in accepted set
            max_unc_accepted = float(self.unc_flat[accept_idx].max())

            results.append({
                "coverage":              float(cov),
                "dice":                  float(dice),
                "auc":                   float(auc_val),
                "error_rate":            error_rate,
                "n_accepted":            n_accept,
                "max_unc_accepted":      max_unc_accepted,
            })

        self.sweep_results = results
        return results

    def to_csv(self, path: str, scenario: str = "") -> None:
        """
        Save the full coverage sweep to risk_coverage.csv.

        Columns: scenario, coverage, dice, auc, error_rate,
                 n_accepted, max_unc_accepted
        """
        if not hasattr(self, "sweep_results"):
            raise RuntimeError("Call sweep_coverage() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "scenario", "coverage", "dice", "auc",
            "error_rate", "n_accepted", "max_unc_accepted",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self.sweep_results:
                writer.writerow({"scenario": scenario, **row})
        print(f"  Saved: {path}")

    def area_under_coverage_curve(self, metric: str = "dice") -> float:
        """
        AUCC: Area Under Coverage-metric Curve.
        Higher = better (model improves more as uncertain pixels are removed).
        """
        cov  = [r["coverage"] for r in self.sweep_results]
        vals = [r[metric]     for r in self.sweep_results]
        return float(np.trapezoid(vals, cov))

    def plot_all(self, output_dir: str, method_name: str = "MC Dropout"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._plot_coverage_curves(output_dir / "selective_coverage.png", method_name)
        self._plot_risk_coverage(output_dir  / "risk_coverage.png",       method_name)

    def _plot_coverage_curves(self, save_path, method_name):
        rs  = self.sweep_results
        cov  = [r["coverage"] for r in rs]
        dice = [r["dice"]     for r in rs]
        auc  = [r["auc"]      for r in rs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)

        ax1.plot(cov, dice, color="#4C72B0", linewidth=2.5, label=method_name)
        ax1.axhline(0.82, color="#E07B54", linestyle="--", linewidth=1.2,
                    label="Dice = 0.82 target")
        ax1.fill_between(cov, dice, 0.82, where=[d >= 0.82 for d in dice],
                         alpha=0.1, color="#4C72B0")
        ax1.set_xlabel("Coverage (fraction of pixels accepted)", fontsize=12)
        ax1.set_ylabel("Dice on accepted pixels", fontsize=12)
        ax1.set_title("Selective Prediction: Coverage vs Dice\n"
                      f"AUCC = {self.area_under_coverage_curve('dice'):.4f}", fontsize=12)
        ax1.legend(fontsize=10); ax1.grid(alpha=0.3)
        ax1.set_xlim(0.1, 1.0)
        ax1.margins(x=0.02, y=0.08)

        ax2.plot(cov, auc, color="#54A87A", linewidth=2.5, label=method_name)
        ax2.axhline(0.98, color="#E07B54", linestyle="--", linewidth=1.2,
                    label="AUC = 0.98 target")
        ax2.set_xlabel("Coverage (fraction of pixels accepted)", fontsize=12)
        ax2.set_ylabel("AUC on accepted pixels", fontsize=12)
        ax2.set_title(f"Coverage vs AUC\n"
                      f"AUCC = {self.area_under_coverage_curve('auc'):.4f}", fontsize=12)
        ax2.legend(fontsize=10); ax2.grid(alpha=0.3)
        ax2.set_xlim(0.1, 1.0)
        ax2.margins(x=0.02, y=0.08)

        plt.suptitle(f"Selective Prediction Curves — {method_name}", fontsize=14)
        style_axes(ax1)
        style_axes(ax2)
        save_figure(fig, save_path)
        print(f"  Saved: {save_path}")

    def _plot_risk_coverage(self, save_path, method_name):
        rs   = self.sweep_results
        cov  = [r["coverage"]   for r in rs]
        risk = [r["error_rate"] for r in rs]

        fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
        ax.plot(cov, risk, color="#E07B54", linewidth=2.5, label=method_name)
        ax.fill_between(cov, risk, alpha=0.15, color="#E07B54")
        ax.set_xlabel("Coverage", fontsize=12)
        ax.set_ylabel("Risk (error rate on accepted pixels)", fontsize=12)
        ax.set_title("Risk-Coverage Curve\n"
                     "Lower risk at lower coverage = uncertainty is useful", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xlim(0.1, 1.0)
        ax.margins(x=0.02, y=0.08)
        style_axes(ax)
        save_figure(fig, save_path)
        print(f"  Saved: {save_path}")

    def run(
        self,
        output_dir: str,
        method_name: str = "MC Dropout",
        scenario: str = "",
        save_csv: bool = True,
    ) -> dict:
        print(f"\n[SelectivePrediction] Sweeping coverage for {method_name}...")
        self.sweep_coverage()
        self.plot_all(output_dir, method_name)

        if save_csv:
            csv_path = Path(output_dir) / "risk_coverage.csv"
            self.to_csv(str(csv_path), scenario=scenario or method_name)

        summary = {
            "method":            method_name,
            "scenario":          scenario,
            "aucc_dice":         self.area_under_coverage_curve("dice"),
            "aucc_auc":          self.area_under_coverage_curve("auc"),
            "full_coverage_dice": self.sweep_results[-1]["dice"],
            "at_90_coverage":    next(
                (r for r in self.sweep_results if r["coverage"] >= 0.9), {}),
        }

        out = Path(output_dir) / "selective_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  AUCC (Dice): {summary['aucc_dice']:.4f}")
        print(f"  AUCC (AUC):  {summary['aucc_auc']:.4f}")
        return summary
