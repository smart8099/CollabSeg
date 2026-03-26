"""Training and evaluation loops for segmentation models."""

from __future__ import annotations

from collections import defaultdict

import torch

from .metrics import binary_segmentation_metrics


def _forward_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Normalize model outputs to a single logits tensor."""
    outputs = model(images)
    if isinstance(outputs, list):
        return outputs[-1]
    return outputs


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
    grad_clip_norm: float | None = None,
    max_batches: int = 0,
) -> dict[str, float]:
    """Train a model for one epoch and return aggregate loss metrics."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = _forward_logits(model, images)
            loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        if max_batches and batch_idx >= max_batches:
            break

    return {"loss": total_loss / max(total_batches, 1)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, float]:
    """Evaluate a model over a data loader and return averaged metrics."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    totals = defaultdict(float)

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = _forward_logits(model, images)
        loss = criterion(logits, masks)
        metrics = binary_segmentation_metrics(logits, masks)

        total_loss += loss.item()
        total_batches += 1
        for key, value in metrics.items():
            totals[key] += value
        if max_batches and batch_idx >= max_batches:
            break

    results = {"loss": total_loss / max(total_batches, 1)}
    for key, value in totals.items():
        results[key] = value / max(total_batches, 1)
    return results
