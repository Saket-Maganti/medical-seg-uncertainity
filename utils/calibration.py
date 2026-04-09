"""
Post-hoc Calibration via Temperature Scaling.

Temperature scaling learns a single scalar T on the validation set
that divides logits before sigmoid: p = sigmoid(logit / T).
T > 1 → softer predictions (less confident) → usually improves ECE.
T < 1 → sharper predictions (more confident).

This is the standard calibration baseline from Guo et al. (2017).
We compare ECE before and after temperature scaling for each method.

Usage:
    from utils.calibration import TemperatureScaler
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)      # learns T on val set
    calibrated_probs = scaler.calibrate(test_logits)
    ece_before, ece_after = scaler.compare_ece(test_probs, calibrated_probs, test_labels)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from utils.figure_style import apply_publication_style, save_figure, style_axes
from utils.metrics import expected_calibration_error, reliability_diagram_data


apply_publication_style()


class TemperatureScaler(nn.Module):
    """
    Learns a single temperature parameter T via NLL minimization on val set.
    Works on raw logits (before sigmoid).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            lr: float = 0.01, max_iter: int = 500, verbose: bool = True):
        """
        Fit temperature on validation logits/labels.

        Args:
            logits: (N,) raw model logits (before sigmoid)
            labels: (N,) binary ground truth
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = self(logits_t)
            loss = criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        T = self.temperature.item()
        if verbose:
            print(f"[TemperatureScaler] Learned T = {T:.4f}")
        return T

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling, return calibrated probabilities."""
        with torch.no_grad():
            logits_t = torch.tensor(logits, dtype=torch.float32)
            scaled   = self(logits_t)
            probs    = torch.sigmoid(scaled).numpy()
        return probs

    def compare_ece(self, probs_before: np.ndarray,
                    probs_after: np.ndarray,
                    labels: np.ndarray,
                    output_dir: str = None) -> dict:
        """
        Compute and compare ECE before/after temperature scaling.
        Optionally save reliability diagram comparison.
        """
        ece_before = expected_calibration_error(probs_before, labels)
        ece_after  = expected_calibration_error(probs_after,  labels)

        result = {
            "temperature":      self.temperature.item(),
            "ece_before":       ece_before,
            "ece_after":        ece_after,
            "ece_improvement":  ece_before - ece_after,
            "improvement_pct":  (ece_before - ece_after) / ece_before * 100,
        }

        print(f"\n[Calibration] Temperature = {result['temperature']:.4f}")
        print(f"  ECE before: {ece_before:.4f}")
        print(f"  ECE after:  {ece_after:.4f}  "
              f"({'↓' if ece_after < ece_before else '↑'}"
              f" {abs(result['improvement_pct']):.1f}%)")

        if output_dir:
            self._plot_reliability_comparison(
                probs_before, probs_after, labels,
                ece_before, ece_after, output_dir)

        return result

    def _plot_reliability_comparison(self, probs_before, probs_after, labels,
                                      ece_before, ece_after, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)

        for ax, probs, ece, title in [
            (axes[0], probs_before, ece_before, "Before temperature scaling"),
            (axes[1], probs_after,  ece_after,  "After temperature scaling"),
        ]:
            centers, conf, acc, counts = reliability_diagram_data(probs, labels)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect")
            ax.bar(centers, acc, width=1.0/len(centers), alpha=0.65,
                   color="#4C72B0", label="Accuracy")
            ax.plot(centers, conf, "ro-", markersize=4, label="Confidence")
            ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=12)
            ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
            ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.margins(x=0.02, y=0.04)
            style_axes(ax)

        fig.suptitle(
            f"Reliability Diagram: Temperature Scaling (T={self.temperature.item():.3f})",
            fontsize=13,
        )
        save_path = Path(output_dir) / "calibration_comparison.png"
        save_figure(fig, save_path, dpi=220)
        print(f"  Saved: {save_path}")


def calibrate_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to already-aggregated probabilities.

    Converts: probs → pseudo-logits → scale by 1/T → sigmoid.
    Appropriate for MC Dropout and TTA where raw per-pass logits are
    unavailable after aggregation.

    Args:
        probs:       (N,) or (H, W) probability array in (0, 1)
        temperature: learned temperature T > 0 (>1 softens, <1 sharpens)

    Returns:
        Calibrated probabilities, same shape as input.
    """
    eps = 1e-7
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(probs_clipped / (1.0 - probs_clipped))   # logit(p)
    scaled = logits / max(float(temperature), 0.05)
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)


def fit_temperature_from_probs(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 500,
    verbose: bool = True,
):
    """
    Fit temperature scaling from aggregated probabilities (not raw logits).

    Converts probs to pseudo-logits first, then optimises T via NLL.
    Appropriate for MC Dropout and TTA where only aggregated probs are available.

    Args:
        val_probs:  (N,) validation probabilities (after aggregation)
        val_labels: (N,) binary ground-truth labels
        lr:         LBFGS learning rate
        max_iter:   max iterations

    Returns:
        (T, scaler) — learned temperature float and fitted TemperatureScaler
    """
    eps = 1e-7
    probs_clipped = np.clip(val_probs, eps, 1.0 - eps)
    pseudo_logits = np.log(probs_clipped / (1.0 - probs_clipped))

    scaler = TemperatureScaler()
    T = scaler.fit(pseudo_logits, val_labels, lr=lr, max_iter=max_iter, verbose=verbose)
    return T, scaler


def fit_temperature_on_model(model, val_loader, device,
                              verbose: bool = True) -> TemperatureScaler:
    """
    Convenience function: collect val logits from a trained model,
    fit and return a TemperatureScaler.
    """
    import torch
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            fov    = batch["fov"].squeeze(1).cpu().numpy()
            logits = model(images).squeeze(1)

            for i in range(len(logits)):
                fov_b = fov[i].astype(bool)
                all_logits.append(logits[i].cpu().numpy()[fov_b])
                all_labels.append(masks[i].squeeze(0).cpu().numpy()[fov_b])

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)

    scaler = TemperatureScaler()
    scaler.fit(logits_np, labels_np, verbose=verbose)
    return scaler
