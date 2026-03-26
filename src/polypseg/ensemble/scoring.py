"""Heuristic feature extraction and scoring for ensemble predictions."""

from __future__ import annotations

import math
import re
from collections import deque

import numpy as np

from .types import PredictionRecord


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract a simple binary boundary map from a binary mask."""
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D.")
    mask = mask.astype(np.uint8)
    eroded = mask.copy()
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return np.clip(mask - eroded, 0, 1)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute a lightweight gradient magnitude map for an RGB image."""
    gray = image.astype(np.float32).mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx**2 + gy**2)


def _component_count(mask: np.ndarray) -> int:
    """Count connected foreground components in a binary mask."""
    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            count += 1
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
    return count


def _pairwise_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the IoU between two binary masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _prompt_features(prompt: str, mask: np.ndarray) -> dict[str, float]:
    """Derive simple prompt-consistency features from text hints and mask geometry."""
    prompt = prompt.strip().lower()
    if not prompt:
        return {"prompt_score": 0.5}

    h, w = mask.shape
    area_ratio = float(mask.mean())
    ys, xs = np.where(mask > 0)
    center_score = 0.5
    if len(xs) > 0:
        cx = float(xs.mean() / max(w - 1, 1))
        cy = float(ys.mean() / max(h - 1, 1))
        dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        center_score = max(0.0, 1.0 - dist / 0.71)

    small_hint = bool(re.search(r"\bsmall\b|\btiny\b", prompt))
    large_hint = bool(re.search(r"\blarge\b|\bbig\b", prompt))
    center_hint = bool(re.search(r"\bcenter\b|\bcentral\b|\bmiddle\b", prompt))

    size_score = 0.5
    if small_hint:
        size_score = 1.0 if area_ratio <= 0.15 else max(0.0, 1.0 - (area_ratio - 0.15) / 0.35)
    elif large_hint:
        size_score = 1.0 if area_ratio >= 0.12 else max(0.0, area_ratio / 0.12)

    prompt_score = size_score
    if center_hint:
        prompt_score = 0.5 * prompt_score + 0.5 * center_score
    return {"prompt_score": float(prompt_score)}


def compute_prediction_features(
    prediction: PredictionRecord,
    image_np: np.ndarray,
    peer_predictions: list[PredictionRecord],
) -> dict[str, float]:
    """Compute heuristic quality features for one prediction against its peers."""
    mask = prediction.mask
    prob_map = prediction.probability_map
    area_ratio = float(mask.mean())
    components = float(_component_count(mask)) if mask.any() else 0.0
    gradient = _gradient_magnitude(image_np)
    boundary = _binary_boundary(mask)
    boundary_strength = float(gradient[boundary > 0].mean() / (gradient.mean() + 1e-6)) if boundary.any() else 0.0
    fg_conf = float(prob_map[mask > 0].mean()) if mask.any() else 0.0
    bg_conf = float(1.0 - prob_map[mask == 0].mean()) if (~mask.astype(bool)).any() else 0.0

    if len(peer_predictions) > 1:
        agreement = np.mean(
            [
                _pairwise_iou(mask, peer.mask)
                for peer in peer_predictions
                if peer.model_name != prediction.model_name
            ]
        )
    else:
        agreement = 1.0

    return {
        "area_ratio": area_ratio,
        "components": components,
        "boundary_strength": boundary_strength,
        "foreground_confidence": fg_conf,
        "background_confidence": bg_conf,
        "agreement_iou": float(agreement),
    }


def score_prediction(
    prediction: PredictionRecord,
    prompt: str,
    image_np: np.ndarray,
    peer_predictions: list[PredictionRecord],
    config: dict,
) -> PredictionRecord:
    """Score one prediction using heuristic quality features and weighted rules."""
    features = compute_prediction_features(prediction, image_np, peer_predictions)
    features.update(_prompt_features(prompt, prediction.mask))

    weights = config["scoring"]["weights"]
    shape_cfg = config["scoring"]["shape"]

    area_ratio = features["area_ratio"]
    min_area = float(shape_cfg["min_area_ratio"])
    max_area = float(shape_cfg["max_area_ratio"])
    shape_area_score = 1.0 if min_area <= area_ratio <= max_area else 0.0
    components_score = max(0.0, 1.0 - features["components"] / max(float(shape_cfg["max_components"]), 1.0))
    shape_score = 0.6 * shape_area_score + 0.4 * components_score

    confidence_score = 0.7 * features["foreground_confidence"] + 0.3 * features["background_confidence"]
    boundary_score = min(max(features["boundary_strength"] / 2.0, 0.0), 1.0)
    agreement_score = features["agreement_iou"]
    prompt_score = features["prompt_score"]

    total_score = (
        float(weights["confidence"]) * confidence_score
        + float(weights["agreement"]) * agreement_score
        + float(weights["shape"]) * shape_score
        + float(weights["boundary"]) * boundary_score
        + float(weights["prompt"]) * prompt_score
    )

    prediction.features = {
        **features,
        "shape_score": float(shape_score),
        "confidence_score": float(confidence_score),
        "boundary_score": float(boundary_score),
        "agreement_score": float(agreement_score),
        "prompt_score": float(prompt_score),
    }
    prediction.score = float(total_score)
    return prediction


def pairwise_consensus_iou(predictions: list[PredictionRecord]) -> float:
    """Compute the mean pairwise IoU across all model predictions."""
    if len(predictions) < 2:
        return 1.0
    scores = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            scores.append(_pairwise_iou(predictions[i].mask, predictions[j].mask))
    return float(np.mean(scores)) if scores else 1.0


def dice_iou(mask: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute Dice and IoU between a predicted mask and ground truth mask."""
    mask = mask.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(mask, target).sum()
    pred_sum = mask.sum()
    target_sum = target.sum()
    union = np.logical_or(mask, target).sum()
    dice = (2 * intersection) / (pred_sum + target_sum + 1e-6)
    iou = intersection / (union + 1e-6)
    return {"dice": float(dice), "iou": float(iou)}
