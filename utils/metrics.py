"""
Evaluation metrics for segmentation + uncertainty quantification.

Segmentation:
  - Dice coefficient (target > 0.82)
  - AUC-ROC (target > 0.98)
  - IoU, Sensitivity, Specificity

Calibration:
  - Expected Calibration Error (ECE) (target < 0.05)

Uncertainty ↔ Error Correlation:
  - AUROC of uncertainty as a predictor of segmentation errors
  - Spearman rank correlation between uncertainty and error masks
  - Patch-level analysis
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr


# ─────────────────────────────────────────────
# Segmentation metrics
# ─────────────────────────────────────────────

def dice_coefficient(preds: np.ndarray, targets: np.ndarray,
                     threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Dice score between binary prediction and ground truth."""
    preds_bin = (preds > threshold).astype(np.float32)
    intersection = (preds_bin * targets).sum()
    return (2.0 * intersection + smooth) / (preds_bin.sum() + targets.sum() + smooth)


def compute_auc(preds: np.ndarray, targets: np.ndarray) -> float:
    """AUC-ROC for vessel probability maps vs ground truth."""
    return roc_auc_score(targets.ravel(), preds.ravel())


def compute_iou(preds: np.ndarray, targets: np.ndarray,
                threshold: float = 0.5, smooth: float = 1e-6) -> float:
    preds_bin = (preds > threshold).astype(np.float32)
    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def compute_sensitivity_specificity(preds: np.ndarray, targets: np.ndarray,
                                    threshold: float = 0.5):
    """Sensitivity (recall) and specificity for binary segmentation."""
    preds_bin = (preds > threshold).astype(np.float32)
    tp = (preds_bin * targets).sum()
    tn = ((1 - preds_bin) * (1 - targets)).sum()
    fp = (preds_bin * (1 - targets)).sum()
    fn = ((1 - preds_bin) * targets).sum()
    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    return float(sensitivity), float(specificity)


# ─────────────────────────────────────────────
# Calibration: ECE
# ─────────────────────────────────────────────

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE).
    Bins predictions by confidence; measures |avg_conf - avg_acc| per bin.
    Lower is better; target < 0.05.

    Args:
        probs:  Predicted probabilities, shape (N,) — flattened pixel probs
        labels: Binary ground truth, shape (N,)
        n_bins: Number of calibration bins
    """
    probs  = probs.ravel()
    labels = labels.ravel().astype(np.float32)
    n      = len(probs)

    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    ece         = 0.0

    for i in range(n_bins):
        lo, hi  = bin_edges[i], bin_edges[i + 1]
        in_bin  = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        n_bin   = in_bin.sum()
        if n_bin == 0:
            continue
        avg_conf = probs[in_bin].mean()
        avg_acc  = labels[in_bin].mean()
        ece     += (n_bin / n) * abs(avg_conf - avg_acc)

    return float(ece)


def reliability_diagram_data(probs: np.ndarray, labels: np.ndarray,
                              n_bins: int = 15):
    """Returns bin centers, mean confidence, and mean accuracy for plotting."""
    probs  = probs.ravel()
    labels = labels.ravel().astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, confidences, accuracies, counts = [], [], [], []

    for i in range(n_bins):
        lo, hi  = bin_edges[i], bin_edges[i + 1]
        in_bin  = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        n_bin   = in_bin.sum()
        if n_bin == 0:
            continue
        centers.append((lo + hi) / 2)
        confidences.append(probs[in_bin].mean())
        accuracies.append(labels[in_bin].mean())
        counts.append(n_bin)

    return np.array(centers), np.array(confidences), np.array(accuracies), np.array(counts)


# ─────────────────────────────────────────────
# Uncertainty ↔ Error Correlation
# ─────────────────────────────────────────────

def uncertainty_error_auroc(uncertainty_map: np.ndarray,
                             error_mask: np.ndarray) -> float:
    """
    AUROC measuring how well uncertainty predicts segmentation errors.
    This is the KEY research metric — proves the model "knows where it fails."

    Args:
        uncertainty_map: epistemic uncertainty (H, W) — higher = more uncertain
        error_mask:      binary error mask (H, W) — 1 where prediction is wrong
    """
    u = uncertainty_map.ravel()
    e = error_mask.ravel().astype(np.int32)
    if e.sum() == 0 or e.sum() == len(e):
        return float("nan")
    return roc_auc_score(e, u)


def uncertainty_error_correlation(uncertainty_map: np.ndarray,
                                   error_mask: np.ndarray,
                                   patch_size: int = 16):
    """
    Patch-level Spearman correlation between mean uncertainty and error rate.
    More robust than pixel-level (avoids spatial autocorrelation).

    Returns:
        rho:   Spearman r  (target: > 0.4 is meaningful)
        pval:  p-value
    """
    H, W     = uncertainty_map.shape
    unc_vals, err_vals = [], []

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            u_patch = uncertainty_map[i:i+patch_size, j:j+patch_size]
            e_patch = error_mask[i:i+patch_size, j:j+patch_size]
            unc_vals.append(u_patch.mean())
            err_vals.append(e_patch.mean())

    rho, pval = spearmanr(unc_vals, err_vals)
    return float(rho), float(pval)


def compute_error_mask(preds: np.ndarray, targets: np.ndarray,
                        threshold: float = 0.5) -> np.ndarray:
    """Binary mask: 1 where prediction disagrees with ground truth."""
    preds_bin = (preds > threshold).astype(np.float32)
    return (preds_bin != targets).astype(np.float32)


# ─────────────────────────────────────────────
# Aggregate evaluation
# ─────────────────────────────────────────────

def evaluate_all(preds: np.ndarray, targets: np.ndarray,
                 uncertainty: np.ndarray = None,
                 fov_mask: np.ndarray = None,
                 threshold: float = 0.5) -> dict:
    """
    Full evaluation suite on a single image or batch.

    Args:
        preds:       Predicted probabilities (H, W) or (B, H, W)
        targets:     Ground truth masks (H, W) or (B, H, W)
        uncertainty: Uncertainty map (H, W) — optional
        fov_mask:    FOV mask to restrict evaluation (H, W) — optional
        threshold:   Binarization threshold

    Returns: dict of metric name → value
    """
    if fov_mask is not None:
        fov = fov_mask.astype(bool)
        preds   = preds[fov]
        targets = targets[fov]
        if uncertainty is not None:
            uncertainty = uncertainty[fov_mask.astype(bool).reshape(uncertainty.shape)]

    metrics = {
        "dice":        dice_coefficient(preds, targets, threshold),
        "auc":         compute_auc(preds, targets),
        "iou":         compute_iou(preds, targets, threshold),
        "ece":         expected_calibration_error(preds, targets),
    }
    sens, spec = compute_sensitivity_specificity(preds, targets, threshold)
    metrics["sensitivity"] = sens
    metrics["specificity"] = spec

    if uncertainty is not None:
        error_mask = compute_error_mask(preds, targets, threshold)
        # Reshape for patch correlation if needed
        if preds.ndim == 1:
            # Already flattened by FOV mask — skip patch correlation
            metrics["unc_auroc"] = uncertainty_error_auroc(
                uncertainty.ravel(), error_mask.ravel())
        else:
            metrics["unc_auroc"] = uncertainty_error_auroc(uncertainty, error_mask)
            rho, pval = uncertainty_error_correlation(uncertainty, error_mask)
            metrics["unc_spearman_r"]    = rho
            metrics["unc_spearman_pval"] = pval

    return metrics
