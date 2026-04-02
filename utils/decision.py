"""
Decision-Theoretic Deferral via Cost-Sensitive Risk Minimization.

The key insight: deferral is a DECISION PROBLEM, not just thresholding.

Standard approach (naive):
    if uncertainty > T → defer          ← arbitrary T, ignores costs

This module (decision-theoretic):
    defer iff E[cost of predicting] > E[cost of deferring]
    → optimal T* emerges from the cost structure, not guesswork

Cost model:
    - c_fp:  cost of false positive (predict vessel, actually background)
    - c_fn:  cost of false negative (miss vessel, actually vessel) ← clinically worse
    - c_def: cost of deferral (clinician review time)

Expected risk at pixel x:
    R_predict(x) = c_fp * P(y=0|x) * p(x) + c_fn * P(y=1|x) * (1 - p(x))
    R_defer(x)   = c_def

Optimal decision:
    defer iff R_predict(x) > R_defer(x)

This is mathematically equivalent to RAG's "answer vs refuse" decision:
    Answer  ↔ predict
    Refuse  ↔ defer
    Hallucination cost ↔ c_fn/c_fp
    Abstention cost    ↔ c_def

References:
    Geifman & El-Yaniv (2017). Selective Classification for Deep Neural Networks.
    Chow (1970). On optimum recognition error and reject tradeoff.
    Hellman (1970). The nearest neighbor classification rule with a reject option.

Usage:
    from utils.decision import DecisionTheoreticalDeferral
    dtd = DecisionTheoreticalDeferral(c_fp=1.0, c_fn=3.0, c_def=1.5)
    decisions = dtd.decide(pred_probs, uncertainty_maps)
    report = dtd.full_analysis(pred_probs, gt_masks, uncertainty_maps)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from pathlib import Path
from typing import Optional, List
import json


class DecisionTheoreticalDeferral:
    """
    Cost-aware deferral policy grounded in statistical decision theory.

    The system minimizes expected risk by comparing:
        R_predict(x) = c_fp * p_bg(x) * pred(x) + c_fn * p_vessel(x) * (1-pred(x))
        R_defer(x)   = c_def

    where p_vessel = predicted probability, p_bg = 1 - p_vessel.

    Args:
        c_fp:     Cost of false positive (predict vessel, actually background)
        c_fn:     Cost of false negative (miss vessel, actually vessel)
                  Typically c_fn > c_fp in clinical settings
        c_def:    Cost of deferral (clinician time, workflow cost)
        use_uncertainty: If True, modulate risk by uncertainty (more principled)
    """

    def __init__(self,
                 c_fp: float = 1.0,
                 c_fn: float = 3.0,
                 c_def: float = 1.5,
                 use_uncertainty: bool = True):
        self.c_fp  = c_fp
        self.c_fn  = c_fn
        self.c_def = c_def
        self.use_uncertainty = use_uncertainty

        # Derived optimal threshold (analytical)
        self.T_analytical = self._compute_analytical_threshold()

    def _compute_analytical_threshold(self) -> float:
        """
        Derive T* analytically from cost parameters.

        When uncertainty = variance of Bernoulli(p):
            Var(p) = p(1-p)

        Defer iff R_predict > R_def:
            c_fp*(1-p)*p + c_fn*p*(1-p) > c_def
            (c_fp + c_fn) * p*(1-p) > c_def
            p*(1-p) > c_def / (c_fp + c_fn)

        This gives a closed-form uncertainty threshold:
            T* = c_def / (c_fp + c_fn)

        This is the exact analog of Bayes-optimal reject option (Chow, 1970).
        """
        T_star = self.c_def / (self.c_fp + self.c_fn)
        return float(T_star)

    def expected_risk_predict(self,
                               pred_prob: np.ndarray,
                               uncertainty: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute expected risk of predicting at each pixel.

        R_predict(x) = c_fp * (1-p) * 1_{p>0.5}   (FP risk)
                     + c_fn * p     * 1_{p<=0.5}   (FN risk)

        With uncertainty modulation:
            R_predict(x) *= (1 + alpha * uncertainty(x))
        This increases risk estimate in high-uncertainty regions.
        """
        p   = pred_prob
        # FP risk: we predict vessel (p > 0.5) but might be wrong
        r_fp = self.c_fp * (1 - p) * (p > 0.5).astype(float)
        # FN risk: we predict background (p <= 0.5) but might miss vessel
        r_fn = self.c_fn * p       * (p <= 0.5).astype(float)
        R    = r_fp + r_fn

        if self.use_uncertainty and uncertainty is not None:
            # Normalize uncertainty to [0,1] for consistent scaling
            u_norm = uncertainty / (uncertainty.max() + 1e-8)
            # Uncertainty amplifies predicted risk
            R = R * (1.0 + u_norm)

        return R

    def decide(self,
               pred_prob: np.ndarray,
               uncertainty: np.ndarray,
               threshold: Optional[float] = None) -> dict:
        """
        Make optimal defer/predict decision for each pixel.

        Args:
            pred_prob:   (H, W) predicted probability
            uncertainty: (H, W) epistemic uncertainty
            threshold:   Override analytical threshold (for sweep)

        Returns:
            dict with:
              'defer_mask':    (H, W) bool — True = defer this pixel
              'R_predict':     (H, W) expected risk of predicting
              'R_defer':       scalar — cost of deferral
              'coverage':      fraction of pixels NOT deferred
              'expected_risk': mean risk over accepted pixels
        """
        T    = threshold if threshold is not None else self.T_analytical
        R_p  = self.expected_risk_predict(pred_prob, uncertainty)
        R_d  = self.c_def

        defer_mask = R_p > R_d               # defer where prediction is too risky
        coverage   = 1.0 - defer_mask.mean()
        E_risk     = float(R_p[~defer_mask].mean()) if (~defer_mask).any() else 0.0

        return {
            "defer_mask":     defer_mask,
            "R_predict":      R_p,
            "R_defer":        R_d,
            "threshold_used": T,
            "coverage":       float(coverage),
            "pct_deferred":   float(defer_mask.mean() * 100),
            "expected_risk":  E_risk,
        }

    def sweep_costs(self,
                    pred_probs: List[np.ndarray],
                    gt_masks:   List[np.ndarray],
                    uncertainties: List[np.ndarray],
                    fov_masks:  Optional[List[np.ndarray]] = None) -> dict:
        """
        Sweep c_fn/c_fp ratio from 1 to 10.
        Shows how optimal deferral rate changes with clinical cost structure.
        This is the key insight: different clinical contexts → different T*.

        Returns:
            dict mapping cost_ratio → {T_star, coverage, dice_accepted, FN_rate}
        """
        fovs    = fov_masks or [None] * len(pred_probs)
        ratios  = np.logspace(0, 1, 20)   # 1 to 10 in log space
        results = {}

        for ratio in ratios:
            # Temporarily update costs
            orig_fn, orig_fp = self.c_fn, self.c_fp
            self.c_fn = ratio * self.c_fp
            T_star    = self._compute_analytical_threshold()

            coverages, fn_rates, dices = [], [], []
            for i in range(len(pred_probs)):
                fov    = fovs[i].astype(bool) if fovs[i] is not None else None
                p, g, u = pred_probs[i], gt_masks[i], uncertainties[i]

                dec = self.decide(p, u, threshold=T_star)
                dm  = dec["defer_mask"]

                if fov is not None:
                    dm_fov = dm[fov]; p_fov = p[fov]; g_fov = g[fov]
                else:
                    dm_fov, p_fov, g_fov = dm.ravel(), p.ravel(), g.ravel()

                coverages.append(1 - dm_fov.mean())

                # Dice on accepted
                acc = ~dm_fov
                if acc.sum() > 0:
                    pb  = (p_fov[acc] > 0.5).astype(float)
                    gv  = g_fov[acc]
                    inter = (pb * gv).sum()
                    dice  = (2*inter + 1e-6) / (pb.sum() + gv.sum() + 1e-6)
                    dices.append(dice)
                    # FN rate: vessel pixels predicted as background
                    vessel_px = gv.sum()
                    fn_rate   = ((1 - pb) * gv).sum() / (vessel_px + 1e-6)
                    fn_rates.append(fn_rate)

            results[float(ratio)] = {
                "T_star":           float(T_star),
                "coverage":         float(np.mean(coverages)),
                "dice_accepted":    float(np.mean(dices))    if dices    else float("nan"),
                "fn_rate":          float(np.mean(fn_rates)) if fn_rates else float("nan"),
                "c_fn_c_fp_ratio":  float(ratio),
            }

            # Restore
            self.c_fn, self.c_fp = orig_fn, orig_fp

        return results

    def compute_bayes_risk(self,
                            pred_probs: List[np.ndarray],
                            gt_masks:   List[np.ndarray],
                            uncertainties: List[np.ndarray],
                            fov_masks:  Optional[List] = None) -> dict:
        """
        Compute actual realized risk under the decision rule.
        Compares:
          - Naive baseline (no deferral)
          - Uncertainty threshold (fixed T)
          - Decision-theoretic (optimal T*)

        Returns breakdown of TP/FP/FN/TN/deferred with associated costs.
        """
        fovs = fov_masks or [None] * len(pred_probs)
        summary = {"naive": [], "fixed_T": [], "decision_theoretic": []}

        for i in range(len(pred_probs)):
            fov = fovs[i].astype(bool) if fovs[i] is not None else None
            p, g, u = pred_probs[i], gt_masks[i], uncertainties[i]

            if fov is not None:
                p, g, u = p[fov], g[fov], u[fov]
            else:
                p, g, u = p.ravel(), g.ravel(), u.ravel()

            p_bin = (p > 0.5).astype(float)

            # Naive: predict everywhere
            fp = (p_bin * (1-g)).sum()
            fn = ((1-p_bin) * g).sum()
            naive_risk = (self.c_fp * fp + self.c_fn * fn) / len(p)
            summary["naive"].append(float(naive_risk))

            # Fixed threshold (median uncertainty)
            T_fixed  = float(np.median(u))
            defer_f  = u >= T_fixed
            acc_f    = ~defer_f
            if acc_f.any():
                fp_f = (p_bin[acc_f] * (1-g[acc_f])).sum()
                fn_f = ((1-p_bin[acc_f]) * g[acc_f]).sum()
                risk_f = (self.c_fp*fp_f + self.c_fn*fn_f + self.c_def*defer_f.sum()) / len(p)
            else:
                risk_f = self.c_def
            summary["fixed_T"].append(float(risk_f))

            # Decision-theoretic: optimal T*
            dec     = self.decide(p.reshape(1, -1)[0].reshape(
                          int(len(p)**0.5), -1) if len(p.shape)==1
                          else p, u)
            defer_d = dec["defer_mask"].ravel() if hasattr(dec["defer_mask"], 'ravel') else u >= self.T_analytical
            acc_d   = ~defer_d
            if acc_d.any():
                fp_d = (p_bin[acc_d] * (1-g[acc_d])).sum()
                fn_d = ((1-p_bin[acc_d]) * g[acc_d]).sum()
                risk_d = (self.c_fp*fp_d + self.c_fn*fn_d + self.c_def*defer_d.sum()) / len(p)
            else:
                risk_d = self.c_def
            summary["decision_theoretic"].append(float(risk_d))

        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in summary.items()}

    def plot_all(self,
                 pred_probs, gt_masks, uncertainties,
                 fov_masks=None, output_dir: str = "results/decision"):
        """Generate all decision-theoretic analysis plots."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Cost sweep
        print("[Decision] Sweeping cost ratios...")
        cost_results = self.sweep_costs(pred_probs, gt_masks, uncertainties, fov_masks)
        self._plot_cost_sensitivity(cost_results, f"{output_dir}/cost_sensitivity.png")

        # 2. Risk comparison
        print("[Decision] Computing Bayes risk...")
        risk_cmp = self.compute_bayes_risk(pred_probs, gt_masks, uncertainties, fov_masks)
        self._plot_risk_comparison(risk_cmp, f"{output_dir}/risk_comparison.png")

        # 3. Decision boundary visualization (first image)
        if len(pred_probs) > 0:
            self._plot_decision_boundary(
                pred_probs[0], gt_masks[0], uncertainties[0],
                fov_masks[0] if fov_masks else None,
                f"{output_dir}/decision_boundary.png")

        # Save summary
        summary = {
            "cost_params":      {"c_fp": self.c_fp, "c_fn": self.c_fn, "c_def": self.c_def},
            "T_analytical":     self.T_analytical,
            "risk_comparison":  risk_cmp,
        }
        with open(f"{output_dir}/decision_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[Decision] Analytical T* = {self.T_analytical:.4f}")
        print(f"  c_fn/c_fp = {self.c_fn/self.c_fp:.1f}x  →  "
              f"defer {self.c_def/(self.c_fp+self.c_fn)*100:.1f}% of uncertain pixels")
        print(f"\n[Decision] Realized risk (mean across test set):")
        for method, vals in risk_cmp.items():
            print(f"  {method:<22}: {vals['mean']:.4f} ± {vals['std']:.4f}")

        return summary

    def _plot_cost_sensitivity(self, cost_results, save_path):
        ratios   = sorted(cost_results.keys())
        T_stars  = [cost_results[r]["T_star"]        for r in ratios]
        cov      = [cost_results[r]["coverage"]       for r in ratios]
        dices    = [cost_results[r]["dice_accepted"]  for r in ratios]
        fn_rates = [cost_results[r]["fn_rate"]        for r in ratios]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].semilogx(ratios, T_stars, "o-", color="#4C72B0", linewidth=2)
        axes[0].set_xlabel("c_fn / c_fp ratio", fontsize=12)
        axes[0].set_ylabel("Optimal threshold T*", fontsize=12)
        axes[0].set_title("How cost ratio shapes T*\n(Higher FN cost → lower threshold → more deferral)", fontsize=11)
        axes[0].grid(alpha=0.3)

        axes[1].semilogx(ratios, cov, "s-", color="#54A87A", linewidth=2, label="Coverage")
        axes[1].semilogx(ratios, dices, "^-", color="#E07B54", linewidth=2, label="Dice (accepted)")
        axes[1].axhline(0.82, color="#E07B54", linestyle="--", linewidth=1, alpha=0.5)
        axes[1].set_xlabel("c_fn / c_fp ratio", fontsize=12)
        axes[1].set_ylabel("Score", fontsize=12)
        axes[1].set_title("Coverage & Dice vs cost ratio", fontsize=11)
        axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

        axes[2].semilogx(ratios, fn_rates, "D-", color="#9B59B6", linewidth=2)
        axes[2].set_xlabel("c_fn / c_fp ratio", fontsize=12)
        axes[2].set_ylabel("FN rate (accepted pixels)", fontsize=12)
        axes[2].set_title("False negative rate vs cost ratio\n(Clinical safety metric)", fontsize=11)
        axes[2].grid(alpha=0.3)

        plt.suptitle("Decision-Theoretic Analysis: Cost Sensitivity\n"
                     f"Base: c_fp={self.c_fp}, c_fn={self.c_fn}, c_def={self.c_def}", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    def _plot_risk_comparison(self, risk_cmp, save_path):
        methods = list(risk_cmp.keys())
        means   = [risk_cmp[m]["mean"] for m in methods]
        stds    = [risk_cmp[m]["std"]  for m in methods]
        colors  = ["#888888", "#4C72B0", "#E07B54"]
        labels  = ["No deferral\n(naive)", "Fixed threshold\n(median unc.)",
                   "Decision-theoretic\n(optimal T*)"]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, means, color=colors, alpha=0.85,
                      yerr=stds, capsize=6, error_kw={"linewidth": 1.5})
        ax.set_ylabel("Expected risk (normalized)", fontsize=12)
        ax.set_title("Risk Reduction via Decision-Theoretic Deferral\n"
                     f"Risk = c_fp×FP + c_fn×FN + c_def×deferred", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        # Annotate % reduction
        base = means[0]
        for i, (bar, m) in enumerate(zip(bars, means)):
            if i > 0:
                pct = (base - m) / base * 100
                ax.text(bar.get_x() + bar.get_width()/2,
                        m + stds[i] + 0.002,
                        f"↓{pct:.1f}%", ha="center", fontsize=10,
                        color="#333333", fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    def _plot_decision_boundary(self, pred_prob, gt_mask, uncertainty,
                                 fov_mask, save_path):
        """4-panel: prediction | R_predict | defer_mask | cost map."""
        R_p = self.expected_risk_predict(pred_prob, uncertainty)
        dec = self.decide(pred_prob, uncertainty)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ["Prediction", "Expected risk R_predict",
                  f"Defer mask (T*={self.T_analytical:.3f})", "Decision cost map"]
        imgs   = [pred_prob, R_p, dec["defer_mask"].astype(float),
                  R_p * dec["defer_mask"] + self.c_def * (~dec["defer_mask"])]
        cmaps  = ["gray", "hot", "RdYlGn_r", "viridis"]

        for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f"Decision-Theoretic Deferral  "
                     f"(c_fp={self.c_fp}, c_fn={self.c_fn}, c_def={self.c_def}  "
                     f"→  T*={self.T_analytical:.4f}  "
                     f"coverage={dec['coverage']:.2f})", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
