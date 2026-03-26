"""Metrics for binary segmentation evaluation."""

from __future__ import annotations

import torch


@torch.no_grad()
def binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute Dice and IoU metrics from logits and binary targets."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    dims = (0, 2, 3)
    intersection = torch.sum(preds * targets, dim=dims)
    pred_sum = torch.sum(preds, dim=dims)
    target_sum = torch.sum(targets, dim=dims)

    dice = ((2 * intersection + eps) / (pred_sum + target_sum + eps)).mean().item()
    iou = ((intersection + eps) / (pred_sum + target_sum - intersection + eps)).mean().item()

    return {"dice": dice, "iou": iou}
