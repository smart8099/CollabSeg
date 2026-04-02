"""Embedding utilities for SegRank artifact generation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _normalize_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    """Compute a normalized histogram vector."""
    hist, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist = hist / total
    return hist


def image_embedding_from_image(image_np: np.ndarray) -> list[float]:
    """Build a compact appearance embedding from one RGB image."""
    image = image_np.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    gray = image.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    grad = np.sqrt(gx**2 + gy**2)

    rgb_mean = image.mean(axis=(0, 1))
    rgb_std = image.std(axis=(0, 1))
    gray_hist = _normalize_histogram(gray, bins=8, value_range=(0.0, 1.0))
    grad_hist = _normalize_histogram(np.clip(grad, 0.0, 1.0), bins=8, value_range=(0.0, 1.0))

    embedding = np.concatenate([rgb_mean, rgb_std, gray_hist, grad_hist], axis=0)
    return embedding.astype(float).tolist()


def morphology_embedding_from_features(features: dict[str, float]) -> list[float]:
    """Convert per-mask morphology features into a retrieval-ready vector."""
    keys = [
        "area_ratio",
        "component_count",
        "compactness",
        "branching_index",
        "volume_distribution_entropy",
        "boundary_ratio",
        "boundary_tortuosity",
        "bbox_aspect_ratio",
        "centroid_x",
        "centroid_y",
        "empty_mask",
    ]
    return [float(features[key]) for key in keys]


def response_embedding_from_evidence(features: dict[str, float]) -> list[float]:
    """Convert per-prediction evidence features into a model-response embedding."""
    keys = [
        "agreement_iou",
        "area_ratio",
        "background_confidence",
        "boundary_strength",
        "component_count",
        "confidence_margin",
        "foreground_confidence",
        "foreground_entropy",
        "mean_entropy",
    ]
    return [float(features[key]) for key in keys]


def aggregate_embedding_vectors(vectors: list[list[float]]) -> dict[str, Any]:
    """Aggregate a list of embedding vectors into mean/std summaries."""
    if not vectors:
        raise ValueError("Cannot aggregate an empty embedding list.")

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "count": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "mean": matrix.mean(axis=0).astype(float).tolist(),
        "std": matrix.std(axis=0).astype(float).tolist(),
        "mean_l2_norm": float(norms.mean()),
        "std_l2_norm": float(norms.std()),
    }


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embedding_distance(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute Euclidean distance between two embedding vectors."""
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)
    return float(math.sqrt(float(np.sum((a - b) ** 2))))
