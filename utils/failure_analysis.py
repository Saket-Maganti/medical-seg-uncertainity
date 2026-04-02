"""
Failure Mode Taxonomy for Retinal Vessel Segmentation.

Classifies segmentation errors into interpretable categories:
  1. Thin vessels     — errors in pixels near vessel centerlines, far from thick vessels
  2. Vessel borders   — errors at the boundary between vessel and background
  3. Low contrast     — errors in image regions with low local contrast
  4. Image periphery  — errors near FOV edges
  5. Crossings/bifurcations — errors at vessel junctions (high curvature)

For each failure mode, computes:
  - Error rate
  - Mean uncertainty in that region
  - Uncertainty-error AUROC within that region

This answers: WHERE does the model fail, and does uncertainty predict it there?

Usage:
    from utils.failure_analysis import FailureModeAnalyzer
    analyzer = FailureModeAnalyzer()
    report = analyzer.analyze(pred, gt, uncertainty, image, fov)
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, thin
from skimage.filters import sobel
from sklearn.metrics import roc_auc_score
from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path


class FailureModeAnalyzer:
    """
    Analyzes segmentation failures by spatial/structural category.
    """

    def analyze_image(self,
                      pred_prob: np.ndarray,
                      gt_mask: np.ndarray,
                      uncertainty: np.ndarray,
                      image_rgb: np.ndarray,
                      fov_mask: Optional[np.ndarray] = None,
                      threshold: float = 0.5) -> dict:
        """
        Full failure mode analysis for a single image.

        Args:
            pred_prob:   (H, W) predicted probability
            gt_mask:     (H, W) binary ground truth
            uncertainty: (H, W) epistemic uncertainty
            image_rgb:   (H, W, 3) original image (for contrast analysis)
            fov_mask:    (H, W) binary FOV mask
            threshold:   binarization threshold

        Returns:
            dict with per-region error rates and uncertainty stats
        """
        pred_bin   = (pred_prob > threshold).astype(np.float32)
        error_mask = (pred_bin != gt_mask).astype(np.float32)
        gt_bin     = gt_mask.astype(bool)

        regions = self._compute_regions(gt_bin, image_rgb, fov_mask)
        results = {}

        for region_name, region_mask in regions.items():
            if region_mask.sum() < 10:
                continue

            region_errors = error_mask[region_mask]
            region_unc    = uncertainty[region_mask]

            error_rate = float(region_errors.mean())
            mean_unc   = float(region_unc.mean())

            # Uncertainty-error AUROC within this region
            if region_errors.sum() > 0 and region_errors.sum() < region_mask.sum():
                try:
                    auroc = float(roc_auc_score(region_errors, region_unc))
                except:
                    auroc = float("nan")
            else:
                auroc = float("nan")

            results[region_name] = {
                "n_pixels":   int(region_mask.sum()),
                "error_rate": error_rate,
                "mean_unc":   mean_unc,
                "unc_auroc":  auroc,
            }

        return results

    def _compute_regions(self, gt_bin: np.ndarray,
                          image_rgb: np.ndarray,
                          fov_mask: Optional[np.ndarray]) -> dict:
        """Compute spatial region masks for failure mode analysis."""
        H, W = gt_bin.shape
        regions = {}

        # 1. Thin vessels: near skeleton but far from thick vessels
        if gt_bin.sum() > 0:
            skeleton  = skeletonize(gt_bin)
            dist_from_skel = ndimage.distance_transform_edt(~skeleton)
            dist_from_bg   = ndimage.distance_transform_edt(gt_bin)
            thin_vessel_mask = gt_bin & (dist_from_bg <= 2)   # vessels ≤ 2px wide
            regions["thin_vessels"] = thin_vessel_mask
        else:
            regions["thin_vessels"] = np.zeros((H, W), dtype=bool)

        # 2. Vessel borders: within 2px of vessel edge
        if gt_bin.sum() > 0:
            dilated  = ndimage.binary_dilation(gt_bin, iterations=2)
            eroded   = ndimage.binary_erosion(gt_bin,  iterations=2)
            border   = dilated & ~eroded
            regions["vessel_borders"] = border
        else:
            regions["vessel_borders"] = np.zeros((H, W), dtype=bool)

        # 3. Low contrast regions: local std of grayscale image
        gray     = image_rgb.mean(axis=2) if image_rgb.ndim == 3 else image_rgb
        local_std = ndimage.generic_filter(gray.astype(np.float32), np.std, size=15)
        low_contrast = local_std < np.percentile(local_std, 25)
        regions["low_contrast"] = low_contrast

        # 4. Image periphery: within 10% of FOV edge
        if fov_mask is not None:
            fov_b   = fov_mask.astype(bool)
            dist_fov = ndimage.distance_transform_edt(fov_b)
            periphery = fov_b & (dist_fov < 0.1 * max(H, W))
        else:
            y_c, x_c = H // 2, W // 2
            Y, X = np.ogrid[:H, :W]
            dist_center = np.sqrt((Y - y_c)**2 + (X - x_c)**2)
            periphery = dist_center > 0.8 * min(H, W) / 2
        regions["periphery"] = periphery

        # 5. Vessel crossings: high curvature in skeleton
        if gt_bin.sum() > 0:
            skeleton = skeletonize(gt_bin)
            # Crossings have more than 2 neighbours in skeleton
            kernel   = np.ones((3, 3), dtype=np.uint8)
            n_neighbours = ndimage.convolve(skeleton.astype(np.uint8),
                                             kernel) * skeleton.astype(np.uint8)
            crossings = (n_neighbours >= 4)   # junction points
            # Dilate slightly to capture surrounding error
            crossings = ndimage.binary_dilation(crossings, iterations=3)
            regions["crossings"] = crossings
        else:
            regions["crossings"] = np.zeros((H, W), dtype=bool)

        # Apply FOV mask to all regions
        if fov_mask is not None:
            fov_b = fov_mask.astype(bool)
            for k in regions:
                regions[k] = regions[k] & fov_b

        return regions

    def analyze_dataset(self, all_preds, all_gts, all_uncertainties,
                         all_images, all_fovs=None) -> dict:
        """
        Run analysis over all test images, aggregate per region.

        Returns:
            dict: region → {error_rate_mean, mean_unc_mean, unc_auroc_mean}
        """
        from collections import defaultdict
        agg = defaultdict(lambda: {"error_rate": [], "mean_unc": [], "unc_auroc": []})

        n = len(all_preds)
        fovs = all_fovs or [None] * n

        for i in range(n):
            result = self.analyze_image(
                all_preds[i], all_gts[i], all_uncertainties[i],
                all_images[i], fovs[i])
            for region, stats in result.items():
                agg[region]["error_rate"].append(stats["error_rate"])
                agg[region]["mean_unc"].append(stats["mean_unc"])
                if not np.isnan(stats["unc_auroc"]):
                    agg[region]["unc_auroc"].append(stats["unc_auroc"])

        summary = {}
        print("\n[FailureAnalysis] Dataset-level failure mode summary:")
        print(f"  {'Region':<22} {'Error Rate':>12} {'Mean Unc':>12} {'Unc AUROC':>12}")
        print("  " + "-" * 60)

        for region, vals in agg.items():
            summary[region] = {
                "error_rate_mean": float(np.mean(vals["error_rate"])),
                "mean_unc_mean":   float(np.mean(vals["mean_unc"])),
                "unc_auroc_mean":  float(np.mean(vals["unc_auroc"])) if vals["unc_auroc"] else float("nan"),
            }
            r = summary[region]
            print(f"  {region:<22} {r['error_rate_mean']:>12.4f} "
                  f"{r['mean_unc_mean']:>12.6f} {r['unc_auroc_mean']:>12.4f}")

        return summary

    def plot_taxonomy(self, summary: dict, output_dir: str):
        """Bar chart of error rates and uncertainty AUROC by failure mode."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        regions   = list(summary.keys())
        err_rates = [summary[r]["error_rate_mean"] for r in regions]
        unc_auroc = [summary[r]["unc_auroc_mean"]  for r in regions]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        colors = ["#4C72B0", "#54A87A", "#E07B54", "#9B59B6", "#E74C3C"]

        ax1.barh(regions, err_rates, color=colors[:len(regions)], alpha=0.8)
        ax1.set_xlabel("Mean Error Rate", fontsize=12)
        ax1.set_title("Error Rate by Failure Mode", fontsize=13)
        ax1.axvline(np.mean(err_rates), color="black", linestyle="--",
                    linewidth=1, label="Overall mean")
        ax1.legend(fontsize=9)

        ax2.barh(regions, unc_auroc, color=colors[:len(regions)], alpha=0.8)
        ax2.set_xlabel("Uncertainty-Error AUROC", fontsize=12)
        ax2.set_title("How Well Uncertainty Predicts Errors\nby Failure Mode", fontsize=13)
        ax2.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Random")
        ax2.legend(fontsize=9)

        plt.suptitle("Failure Mode Taxonomy", fontsize=14)
        plt.tight_layout()
        save_path = Path(output_dir) / "failure_taxonomy.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
