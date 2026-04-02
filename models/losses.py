"""
Loss functions for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).reshape(-1)
        targets = targets.reshape(-1)
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss.
    BCE handles per-pixel accuracy; Dice handles global overlap.
    alpha controls the balance (default 0.5 / 0.5).
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0, pos_weight: float = 4.0):
        super().__init__()
        self.alpha      = alpha
        self.dice_loss  = DiceLoss(smooth=smooth)
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        self.bce_loss   = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        return (self.alpha * bce +
                (1 - self.alpha) * self.dice_loss(logits, targets))


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced segmentation.
    Useful if Dice+BCE still under-segments thin vessels.
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce_exp = torch.exp(-bce)
        focal   = self.alpha * (1 - bce_exp) ** self.gamma * bce
        return focal.mean()


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_alpha: float = 0.8,
        focal_gamma: float = 2.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.dice_weight * dice + (1 - self.dice_weight) * focal


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.33,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(logits, targets)
        return torch.pow(tversky_loss.clamp(min=1e-6), self.gamma)


def build_loss(
    loss_name: str,
    alpha: float = 0.5,
    pos_weight: float = 4.0,
    focal_alpha: float = 0.8,
    focal_gamma: float = 2.0,
    tversky_alpha: float = 0.7,
    tversky_beta: float = 0.3,
    tversky_gamma: float = 1.33,
):
    loss_name = loss_name.lower()
    if loss_name == "dice_bce":
        return DiceBCELoss(alpha=alpha, pos_weight=pos_weight)
    if loss_name == "dice_focal":
        return DiceFocalLoss(
            dice_weight=alpha,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
    if loss_name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
        )
    raise ValueError(f"Unknown loss '{loss_name}'. Expected one of: dice_bce, dice_focal, focal_tversky")
