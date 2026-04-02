"""
Evidential Deep Learning (EDL) for Uncertainty Quantification.

Instead of a single sigmoid output, EDL outputs parameters of a
Dirichlet distribution over the class probabilities.
Uncertainty is derived from the Dirichlet concentration:
  - High evidence (concentrated Dirichlet) → low uncertainty
  - Low evidence (diffuse Dirichlet) → high uncertainty

Separate epistemic (model) and aleatoric (data) uncertainty.

References:
    Sensoy et al. (2018). Evidential Deep Learning to Quantify
    Classification Uncertainty. NeurIPS 2018.

Usage:
    model = EDLUNet()
    output = model(x)  # returns alpha (Dirichlet params)
    uncertainty = edl_uncertainty(alpha)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class EDLUNet(nn.Module):
    """
    U-Net with Evidential output head.
    Output: 2-channel alpha parameters of a Dirichlet distribution
            (one per class: background, vessel).
    Uncertainty = K / sum(alpha) where K=2 classes.
    """

    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: str = "imagenet",
                 in_channels: int = 3):
        super().__init__()

        # U-Net with 2-channel output (Dirichlet alpha parameters)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=2,          # alpha_0 (background), alpha_1 (vessel)
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns:
            dict with:
              'alpha':       Dirichlet parameters (B, 2, H, W), all > 1
              'prob':        Expected probability of vessel class
              'uncertainty': Total uncertainty (vacuity)
              'aleatoric':   Data uncertainty
              'epistemic':   Model uncertainty
        """
        logits = self.unet(x)

        # Ensure alpha > 1 (softplus + 1)
        alpha  = F.softplus(logits) + 1.0     # (B, 2, H, W)
        S      = alpha.sum(dim=1, keepdim=True)  # Dirichlet strength

        # Predicted probability (expected value of Dirichlet)
        prob   = alpha[:, 1:2, :, :] / S      # vessel probability

        # Vacuity (epistemic uncertainty): K / S
        K      = torch.tensor(2.0, device=x.device)
        vacuity = K / S

        # Dissonance (aleatoric): disagreement between evidence
        # Simplified: variance of the Dirichlet
        aleatoric = (alpha * (S - alpha)) / (S * S * (S + 1))
        aleatoric = aleatoric.sum(dim=1, keepdim=True)

        return {
            "alpha":       alpha,
            "prob":        prob,
            "uncertainty": vacuity,       # use as epistemic uncertainty
            "aleatoric":   aleatoric,
            "epistemic":   vacuity,
            "logits":      logits,
        }


# ── EDL Loss ──────────────────────────────────────────────────────────────────

class EDLLoss(nn.Module):
    """
    Evidential loss for segmentation.
    Combines:
      1. Expected cross-entropy over Dirichlet
      2. KL divergence regularization (penalizes uncertain predictions)
    """

    def __init__(self, lambda_kl: float = 0.1, annealing_epochs: int = 10):
        super().__init__()
        self.lambda_kl        = lambda_kl
        self.annealing_epochs = annealing_epochs

    def forward(self, alpha: torch.Tensor,
                targets: torch.Tensor,
                epoch: int = 0) -> torch.Tensor:
        """
        Args:
            alpha:   Dirichlet parameters (B, 2, H, W)
            targets: Binary ground truth (B, 1, H, W)
            epoch:   Current epoch (for KL annealing)
        """
        # One-hot targets
        targets_oh = torch.cat([1 - targets, targets], dim=1).float()  # (B, 2, H, W)

        S    = alpha.sum(dim=1, keepdim=True)
        K    = alpha.shape[1]

        # Expected cross-entropy: E[log p(y | theta)]
        # = psi(alpha_k) - psi(S)  where psi = digamma
        log_likelihood = (targets_oh * (
            torch.digamma(alpha) - torch.digamma(S)
        )).sum(dim=1).mean()
        nll_loss = -log_likelihood

        # KL divergence from Dirichlet(alpha_tilde) to Dirichlet(1)
        # alpha_tilde = targets_oh + (1 - targets_oh) * alpha (remove target evidence)
        alpha_tilde = targets_oh + (1.0 - targets_oh) * alpha
        kl_loss     = self._kl_divergence(alpha_tilde, K)

        # Annealing: ramp up KL penalty
        annealing_coef = min(1.0, epoch / self.annealing_epochs)
        total_loss = nll_loss + self.lambda_kl * annealing_coef * kl_loss

        return total_loss

    def _kl_divergence(self, alpha: torch.Tensor, K: int) -> torch.Tensor:
        """KL(Dir(alpha) || Dir(1,...,1))"""
        ones    = torch.ones_like(alpha)
        S_alpha = alpha.sum(dim=1, keepdim=True)
        S_ones  = torch.tensor(float(K), device=alpha.device)

        kl = (torch.lgamma(S_alpha) - torch.lgamma(S_ones)
              - torch.lgamma(alpha).sum(dim=1, keepdim=True)
              + (alpha - 1) * (torch.digamma(alpha)
                               - torch.digamma(S_alpha))).sum(dim=1).mean()
        return kl


# ── Aleatoric uncertainty head ─────────────────────────────────────────────────

class AleatoricUNet(nn.Module):
    """
    U-Net with heteroscedastic (aleatoric) uncertainty head.
    Outputs both mean prediction and log-variance.
    Trained with Gaussian NLL loss.

    Separate from EDL — captures data noise, not model uncertainty.
    """

    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: str = "imagenet",
                 in_channels: int = 3):
        super().__init__()

        # Shared encoder
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=2,           # mean + log_var
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> dict:
        out      = self.unet(x)
        mean     = out[:, 0:1, :, :]    # segmentation logit
        log_var  = out[:, 1:2, :, :]    # log aleatoric variance
        var      = torch.exp(log_var)
        prob     = torch.sigmoid(mean)

        return {
            "mean":       mean,
            "prob":       prob,
            "log_var":    log_var,
            "aleatoric":  var,
        }


class AleatoricLoss(nn.Module):
    """
    Gaussian NLL loss for aleatoric uncertainty learning.
    L = 0.5 * exp(-log_var) * BCE(mean, target) + 0.5 * log_var
    """

    def __init__(self):
        super().__init__()

    def forward(self, mean: torch.Tensor,
                log_var: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy_with_logits(mean, targets, reduction="none")
        loss    = 0.5 * torch.exp(-log_var) * bce + 0.5 * log_var
        return loss.mean()
