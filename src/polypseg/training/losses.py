"""Loss functions used for binary polyp segmentation training."""

from __future__ import annotations

import torch
import torch.nn as nn


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft Dice loss directly from logits and binary targets."""
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted combination of BCE-with-logits loss and Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        """Initialize the composite loss with configurable weights."""
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted BCE-plus-Dice training loss."""
        bce = self.bce(logits, targets)
        dice = dice_loss_from_logits(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
