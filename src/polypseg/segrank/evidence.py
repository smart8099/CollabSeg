"""Prediction evidence utilities for SegRank offline artifacts."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .embeddings import response_embedding_from_evidence
from .types import EvidenceSummary


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract a simple binary boundary map from a binary mask."""
    binary = mask.astype(np.uint8)
    eroded = binary.copy()
    eroded[1:-1, 1:-1] = (
        binary[1:-1, 1:-1]
        & binary[:-2, 1:-1]
        & binary[2:, 1:-1]
        & binary[1:-1, :-2]
        & binary[1:-1, 2:]
    )
    return np.clip(binary - eroded, 0, 1)


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
    binary = mask.astype(bool)
    visited = np.zeros_like(binary, dtype=bool)
    count = 0
    height, width = binary.shape
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            count += 1
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
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


def compute_prediction_evidence(
    prediction: Any,
    image_np: np.ndarray,
    peer_predictions: list[Any],
    prompt: str = "",
) -> dict[str, float]:
    """Compute reusable evidence features for one prediction."""
    mask = prediction.mask
    prob_map = prediction.probability_map
    gradient = _gradient_magnitude(image_np)
    boundary = _binary_boundary(mask)
    agreement = 1.0
    if len(peer_predictions) > 1:
        agreement = float(
            np.mean(
                [
                    _pairwise_iou(mask, peer.mask)
                    for peer in peer_predictions
                    if peer.model_name != prediction.model_name
                ]
            )
        )
    features = {
        "area_ratio": float(mask.mean()),
        "component_count": float(_component_count(mask)) if mask.any() else 0.0,
        "boundary_strength": float(gradient[boundary > 0].mean() / (gradient.mean() + 1e-6)) if boundary.any() else 0.0,
        "foreground_confidence": float(prob_map[mask > 0].mean()) if mask.any() else 0.0,
        "background_confidence": float(1.0 - prob_map[mask == 0].mean()) if (~mask.astype(bool)).any() else 0.0,
        "agreement_iou": agreement,
    }

    prob_map = np.clip(prediction.probability_map.astype(np.float32), 1e-6, 1.0 - 1e-6)
    entropy_map = -(prob_map * np.log(prob_map) + (1.0 - prob_map) * np.log(1.0 - prob_map))
    mask = prediction.mask.astype(bool)

    features.update(
        {
            "mean_entropy": float(entropy_map.mean()),
            "foreground_entropy": float(entropy_map[mask].mean()) if mask.any() else 0.0,
            "confidence_margin": float(np.abs(prob_map - 0.5).mean()),
            "prompt_used": float(bool(prompt.strip())),
        }
    )
    features["embedding"] = response_embedding_from_evidence(features)
    return features


def aggregate_evidence(features: list[dict[str, float]]) -> EvidenceSummary:
    """Aggregate evidence feature dictionaries into mean/std summaries."""
    if not features:
        raise ValueError("Cannot aggregate an empty evidence feature list.")

    keys = sorted(key for key in features[0].keys() if key != "embedding")
    means = {key: float(np.mean([item[key] for item in features])) for key in keys}
    stds = {key: float(np.std([item[key] for item in features])) for key in keys}
    return EvidenceSummary(
        num_samples=len(features),
        feature_means=means,
        feature_stds=stds,
    )
