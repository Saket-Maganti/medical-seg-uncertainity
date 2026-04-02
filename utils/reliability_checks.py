"""
Reliability Checks for Uncertainty Estimation.

Answers the critical question: "Can we TRUST this uncertainty?"

Three checks:
  1. Noise sensitivity:
     Add Gaussian noise → uncertainty should increase monotonically.
     If it doesn't, the uncertainty estimate is not detecting input perturbations.

  2. Out-of-distribution detection:
     Cross-dataset images (STARE/CHASE) should have higher mean uncertainty
     than in-distribution (DRIVE) images.
     Checks whether uncertainty captures domain shift.

  3. Overconfident failure detection:
     Identify pixels where: uncertainty is LOW but prediction is WRONG.
     These are the dangerous cases — the model is wrong AND confident.
     A good uncertainty system minimizes this region.

All three checks produce a boolean verdict: PASS / FAIL / WARN.
If your system passes all three, you can defensibly claim trustworthy uncertainty.

Usage:
    from utils.reliability_checks import ReliabilityChecker
    checker = ReliabilityChecker(model, device)
    report = checker.run_all(test_loader, output_dir="results/reliability")
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, mannwhitneyu, ttest_ind
from pathlib import Path
from typing import List, Optional
import json
from utils.mc_dropout import mc_dropout_predict


class ReliabilityChecker:
    """
    Runs three systematic reliability checks on an uncertainty estimator.

    Args:
        model:          Trained MCDropoutUNet (eval mode)
        device:         torch.device
        n_passes:       MC Dropout forward passes
        noise_levels:   Gaussian noise std values for sensitivity check
    """

    def __init__(self, model, device,
                 n_passes: int = 20,
                 noise_levels: List[float] = None):
        self.model        = model
        self.device       = device
        self.n_passes     = n_passes
        self.noise_levels = noise_levels or [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    # ── Check 1: Noise Sensitivity ────────────────────────────────────────────

    def check_noise_sensitivity(self,
                                 images: torch.Tensor,
                                 output_dir: str) -> dict:
        """
        Inject Gaussian noise at increasing levels.
        Expected behaviour: mean uncertainty increases monotonically with noise.
        Pass criterion: Spearman ρ(noise_level, mean_uncertainty) > 0.9

        This validates that the uncertainty estimate responds to
        real degradation in input quality — a basic sanity check.
        """
        print("\n[Reliability] Check 1: Noise sensitivity...")
        results = {}
        mean_uncs = []

        for sigma in self.noise_levels:
            if sigma == 0.0:
                noisy = images.clone()
            else:
                noise = torch.randn_like(images) * sigma
                noisy = (images + noise).clamp(-3.0, 3.0)  # stay in normalized range

            noisy = noisy.to(self.device)
            _, variance = mc_dropout_predict(self.model, noisy, T=self.n_passes)
            mean_unc = float(variance.mean().item())
            mean_uncs.append(mean_unc)

            results[float(sigma)] = {
                "mean_uncertainty": mean_unc,
                "mean_pred":        float("nan"),
            }
            print(f"  σ={sigma:.2f}: mean_unc={mean_unc:.6f}")

        # Spearman correlation between noise level and uncertainty
        rho, pval = spearmanr(self.noise_levels, mean_uncs)

        verdict = "PASS" if rho > 0.9 else ("WARN" if rho > 0.6 else "FAIL")
        print(f"  Spearman ρ = {rho:.3f}  (p={pval:.4f})  →  {verdict}")

        summary = {
            "per_level":   results,
            "spearman_rho": float(rho),
            "spearman_p":   float(pval),
            "verdict":      verdict,
            "interpretation": (
                "Uncertainty increases with noise as expected." if verdict == "PASS"
                else "Uncertainty partially tracks noise." if verdict == "WARN"
                else "DANGER: Uncertainty does not respond to input degradation."
            ),
        }
        self._plot_noise_sensitivity(results, rho, output_dir)
        return summary

    # ── Check 2: OOD Detection ────────────────────────────────────────────────

    def check_ood_detection(self,
                             id_uncertainties: List[float],
                             ood_uncertainties: List[float],
                             id_name: str = "DRIVE",
                             ood_name: str = "STARE",
                             output_dir: str = None) -> dict:
        """
        Test whether OOD images have higher uncertainty than in-distribution.
        Pass criterion:
          - Mann-Whitney U test: p < 0.05 (OOD > ID in uncertainty)
          - Mean OOD uncertainty > mean ID uncertainty

        If the model is overconfident on OOD data, this check will FAIL —
        which itself is a finding worth reporting.
        """
        print(f"\n[Reliability] Check 2: OOD detection ({id_name} vs {ood_name})...")

        id_arr  = np.array(id_uncertainties)
        ood_arr = np.array(ood_uncertainties)

        mean_id  = float(id_arr.mean())
        mean_ood = float(ood_arr.mean())

        # One-sided Mann-Whitney: is OOD > ID?
        stat, pval = mannwhitneyu(ood_arr, id_arr, alternative="greater")
        direction  = "OOD > ID" if mean_ood > mean_id else "OOD ≤ ID"

        if pval < 0.05 and mean_ood > mean_id:
            verdict = "PASS"
        elif mean_ood > mean_id:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        print(f"  Mean unc [{id_name}]:   {mean_id:.6f}")
        print(f"  Mean unc [{ood_name}]:  {mean_ood:.6f}  ({direction})")
        print(f"  Mann-Whitney p = {pval:.4f}  →  {verdict}")

        summary = {
            "mean_id_uncertainty":  mean_id,
            "mean_ood_uncertainty": mean_ood,
            "ratio_ood_to_id":      float(mean_ood / (mean_id + 1e-10)),
            "mannwhitney_stat":     float(stat),
            "mannwhitney_p":        float(pval),
            "verdict":              verdict,
            "interpretation": (
                f"OOD images ({ood_name}) have significantly higher uncertainty. "
                "Model is appropriately uncertain about unseen distribution." if verdict == "PASS"
                else f"OOD uncertainty trend exists but not significant." if verdict == "WARN"
                else f"DANGER: Model is NOT more uncertain on OOD data ({ood_name}). "
                     "May be overconfident on unseen distributions."
            ),
        }

        if output_dir:
            self._plot_ood_comparison(id_arr, ood_arr, id_name, ood_name,
                                       verdict, output_dir)
        return summary

    # ── Check 3: Overconfident Failures ───────────────────────────────────────

    def check_overconfident_failures(self,
                                      pred_probs: List[np.ndarray],
                                      gt_masks: List[np.ndarray],
                                      uncertainties: List[np.ndarray],
                                      fov_masks: Optional[List] = None,
                                      low_unc_pct: float = 10.0,
                                      output_dir: str = None) -> dict:
        """
        Identify overconfident failures:
        pixels where uncertainty is in the bottom X% BUT prediction is wrong.

        These are the MOST DANGEROUS predictions — the model is confident AND wrong.
        Pass criterion: overconfident failure rate < 2× overall error rate.

        Args:
            low_unc_pct: define "low uncertainty" as bottom X percentile
        """
        print(f"\n[Reliability] Check 3: Overconfident failures "
              f"(low unc = bottom {low_unc_pct}%)...")

        fovs = fov_masks or [None] * len(pred_probs)
        all_preds, all_gts, all_uncs = [], [], []

        for i in range(len(pred_probs)):
            fov = fovs[i].astype(bool) if fovs[i] is not None else None
            p, g, u = pred_probs[i], gt_masks[i], uncertainties[i]
            if fov is not None:
                p, g, u = p[fov], g[fov], u[fov]
            all_preds.append(p.ravel())
            all_gts.append(g.ravel())
            all_uncs.append(u.ravel())

        preds_flat = np.concatenate(all_preds)
        gts_flat   = np.concatenate(all_gts).astype(float)
        uncs_flat  = np.concatenate(all_uncs)

        # Overall error rate
        errors_flat    = ((preds_flat > 0.5).astype(float) != gts_flat).astype(float)
        overall_err    = float(errors_flat.mean())

        # Low-uncertainty pixels
        unc_threshold  = np.percentile(uncs_flat, low_unc_pct)
        low_unc_mask   = uncs_flat <= unc_threshold

        # Error rate within low-uncertainty pixels
        low_unc_err    = float(errors_flat[low_unc_mask].mean())
        n_overconf_fail = int((errors_flat[low_unc_mask]).sum())

        ratio          = low_unc_err / (overall_err + 1e-8)

        if ratio < 1.0:
            verdict = "PASS"   # low-unc pixels have FEWER errors than average
        elif ratio < 2.0:
            verdict = "WARN"
        else:
            verdict = "FAIL"   # low-unc pixels have 2× more errors — dangerous

        print(f"  Overall error rate:           {overall_err:.4f}")
        print(f"  Error rate (low-unc pixels):  {low_unc_err:.4f}  "
              f"(ratio={ratio:.2f}×)  →  {verdict}")
        print(f"  Overconfident failures:        {n_overconf_fail:,} pixels")

        summary = {
            "overall_error_rate":     overall_err,
            "low_unc_error_rate":     low_unc_err,
            "overconfident_ratio":    float(ratio),
            "n_overconfident_failures": n_overconf_fail,
            "unc_threshold_used":     float(unc_threshold),
            "low_unc_percentile":     low_unc_pct,
            "verdict":                verdict,
            "interpretation": (
                "Low-uncertainty predictions are MORE accurate than average. "
                "Uncertainty correctly identifies reliable pixels." if verdict == "PASS"
                else "Slight overconfidence present — monitor in deployment." if verdict == "WARN"
                else "DANGER: Low-uncertainty predictions have disproportionate errors. "
                     "Uncertainty is INVERTING signal in some regions."
            ),
        }

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self._plot_overconfident_failures(
                preds_flat, gts_flat, uncs_flat, errors_flat,
                unc_threshold, verdict, output_dir)

        return summary

    # ── Run all checks ────────────────────────────────────────────────────────

    def run_all(self, test_loader,
                id_uncertainties: Optional[List[float]] = None,
                ood_uncertainties: Optional[List[float]] = None,
                pred_probs: Optional[List] = None,
                gt_masks: Optional[List] = None,
                uncertainties: Optional[List] = None,
                fov_masks: Optional[List] = None,
                ood_name: str = "STARE",
                output_dir: str = "results/reliability") -> dict:
        """
        Run all reliability checks and produce a summary verdict.

        Args:
            test_loader:       DataLoader for noise sensitivity check
            id_uncertainties:  Per-image mean uncertainties (in-distribution)
            ood_uncertainties: Per-image mean uncertainties (OOD)
            pred_probs/gt_masks/uncertainties: For overconfident failure check
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        full_report = {}

        # Check 1: Noise sensitivity
        print("[Reliability] Running Check 1...")
        batch = next(iter(test_loader))
        images_sample = batch["image"][:4]  # use first 4 images
        full_report["check1_noise"] = self.check_noise_sensitivity(
            images_sample, output_dir)

        # Check 2: OOD detection (if data provided)
        if id_uncertainties is not None and ood_uncertainties is not None:
            full_report["check2_ood"] = self.check_ood_detection(
                id_uncertainties, ood_uncertainties,
                ood_name=ood_name, output_dir=output_dir)
        else:
            print("\n[Reliability] Check 2: Skipped (run cross_dataset.py first "
                  "to get OOD uncertainties)")
            full_report["check2_ood"] = {"verdict": "SKIPPED"}

        # Check 3: Overconfident failures
        if pred_probs is not None:
            full_report["check3_overconfidence"] = self.check_overconfident_failures(
                pred_probs, gt_masks, uncertainties, fov_masks, output_dir=output_dir)
        else:
            print("\n[Reliability] Check 3: Skipped (run evaluate.py first)")
            full_report["check3_overconfidence"] = {"verdict": "SKIPPED"}

        # Overall verdict
        verdicts = [v.get("verdict", "SKIPPED") for v in full_report.values()]
        if "FAIL"  in verdicts:  overall = "FAIL"
        elif "WARN" in verdicts: overall = "WARN"
        elif "PASS" in verdicts: overall = "PASS"
        else:                    overall = "SKIPPED"

        full_report["overall_verdict"] = overall

        print(f"\n{'='*50}")
        print(f"RELIABILITY CHECKS SUMMARY")
        print(f"{'='*50}")
        for check, result in full_report.items():
            if check == "overall_verdict":
                continue
            v = result.get("verdict", "—")
            emoji = "✓" if v == "PASS" else ("⚠" if v == "WARN"
                    else ("✗" if v == "FAIL" else "—"))
            print(f"  {emoji} {check}: {v}")
        print(f"{'='*50}")
        print(f"  OVERALL: {overall}")
        print(f"{'='*50}\n")

        with open(f"{output_dir}/reliability_report.json", "w") as f:
            json.dump(full_report, f, indent=2)

        self._plot_summary_dashboard(full_report, output_dir)
        return full_report

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _plot_noise_sensitivity(self, results, rho, output_dir):
        sigmas = sorted(results.keys())
        uncs   = [results[s]["mean_uncertainty"] for s in sigmas]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sigmas, uncs, "o-", color="#4C72B0", linewidth=2.5, markersize=8)
        ax.fill_between(sigmas, uncs, alpha=0.15, color="#4C72B0")
        ax.set_xlabel("Gaussian noise σ", fontsize=12)
        ax.set_ylabel("Mean epistemic uncertainty", fontsize=12)
        ax.set_title(f"Noise Sensitivity Check\n"
                     f"Spearman ρ = {rho:.3f}  "
                     f"({'PASS ✓' if rho > 0.9 else 'WARN ⚠' if rho > 0.6 else 'FAIL ✗'})",
                     fontsize=12)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/noise_sensitivity.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_ood_comparison(self, id_arr, ood_arr, id_name, ood_name,
                              verdict, output_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.hist(id_arr,  bins=15, alpha=0.6, color="#4C72B0", label=id_name,  density=True)
        ax1.hist(ood_arr, bins=15, alpha=0.6, color="#E07B54", label=ood_name, density=True)
        ax1.set_xlabel("Mean uncertainty per image", fontsize=11)
        ax1.set_ylabel("Density", fontsize=11)
        ax1.set_title(f"OOD Detection: {id_name} vs {ood_name}\n"
                      f"Verdict: {verdict}", fontsize=11)
        ax1.legend()

        ax2.boxplot([id_arr, ood_arr], labels=[id_name, ood_name],
                    patch_artist=True,
                    boxprops=dict(facecolor="#E6F1FB"),
                    medianprops=dict(color="#185FA5", linewidth=2))
        ax2.set_ylabel("Mean uncertainty", fontsize=11)
        ax2.set_title("Distribution comparison", fontsize=11)
        ax2.grid(axis="y", alpha=0.3)

        plt.suptitle("Check 2: OOD Detection via Uncertainty", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ood_detection.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_overconfident_failures(self, preds, gts, uncs, errors,
                                      unc_threshold, verdict, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Scatter: uncertainty vs error (sample for speed)
        idx = np.random.choice(len(preds), min(5000, len(preds)), replace=False)
        axes[0].scatter(uncs[idx], errors[idx], alpha=0.3, s=2, color="#4C72B0")
        axes[0].axvline(unc_threshold, color="red", linestyle="--",
                        linewidth=1.5, label=f"Low-unc threshold (p{10})")
        axes[0].set_xlabel("Epistemic uncertainty", fontsize=11)
        axes[0].set_ylabel("Error (0=correct, 1=wrong)", fontsize=11)
        axes[0].set_title("Uncertainty vs prediction error\n"
                          "Ideal: errors concentrate at HIGH uncertainty", fontsize=11)
        axes[0].legend()

        # Bar: error rate by uncertainty quartile
        quartiles = np.percentile(uncs, [0, 25, 50, 75, 100])
        labels, err_rates = [], []
        for i in range(4):
            lo, hi = quartiles[i], quartiles[i+1]
            mask = (uncs >= lo) & (uncs < hi)
            if mask.sum() > 0:
                labels.append(f"Q{i+1}\n[{lo:.3f},{hi:.3f}]")
                err_rates.append(float(errors[mask].mean()))

        colors = ["#4C72B0", "#54A87A", "#E07B54", "#E74C3C"]
        axes[1].bar(labels, err_rates, color=colors[:len(labels)], alpha=0.85)
        axes[1].axhline(float(errors.mean()), color="black",
                        linestyle="--", linewidth=1.5, label="Overall error rate")
        axes[1].set_ylabel("Error rate", fontsize=11)
        axes[1].set_title(f"Error rate by uncertainty quartile\n"
                          f"Verdict: {verdict}", fontsize=11)
        axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

        plt.suptitle("Check 3: Overconfident Failure Analysis", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overconfident_failures.png", dpi=150,
                    bbox_inches="tight")
        plt.close()

    def _plot_summary_dashboard(self, report, output_dir):
        """Single-page reliability dashboard."""
        fig = plt.figure(figsize=(12, 4))
        checks = {
            "Noise\nSensitivity":      report.get("check1_noise", {}),
            "OOD\nDetection":          report.get("check2_ood",   {}),
            "Overconfident\nFailures": report.get("check3_overconfidence", {}),
        }
        overall = report.get("overall_verdict", "—")

        color_map = {"PASS": "#54A87A", "WARN": "#E07B54",
                     "FAIL": "#E74C3C", "SKIPPED": "#AAAAAA"}

        ax = fig.add_subplot(111)
        ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
        ax.set_title(f"Reliability Dashboard  |  Overall: {overall}",
                     fontsize=14, fontweight="bold", pad=15)

        for i, (name, result) in enumerate(checks.items()):
            v     = result.get("verdict", "SKIPPED")
            color = color_map.get(v, "#AAAAAA")
            rect  = plt.Rectangle((i * 1.2 + 0.1, 0.3), 1.0, 1.4,
                                   facecolor=color, alpha=0.85, linewidth=2,
                                   edgecolor="white")
            ax.add_patch(rect)
            ax.text(i * 1.2 + 0.6, 1.2, v, ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")
            ax.text(i * 1.2 + 0.6, 0.6, name, ha="center", va="center",
                    fontsize=11, color="white")
            interp = result.get("interpretation", "")
            if interp:
                ax.text(i * 1.2 + 0.6, 0.15,
                        interp[:50] + ("..." if len(interp) > 50 else ""),
                        ha="center", va="center", fontsize=7, color="#333333",
                        wrap=True)

        # Overall verdict box
        oc = color_map.get(overall, "#AAAAAA")
        rect_ov = plt.Rectangle((3.7, 0.3), 0.25, 1.4, facecolor=oc,
                                  alpha=0.9, linewidth=2, edgecolor="white")
        ax.add_patch(rect_ov)
        ax.text(3.825, 1.0, overall, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", rotation=90)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/reliability_dashboard.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_dir}/reliability_dashboard.png")
