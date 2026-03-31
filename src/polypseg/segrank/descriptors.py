"""Dataset-level image descriptor utilities for SegRank."""

from __future__ import annotations

import math

import numpy as np

from .types import DatasetDescriptor


def compute_image_descriptor(image_np: np.ndarray) -> dict[str, float | list[float]]:
    """Compute a lightweight descriptor from one RGB image."""
    image = image_np.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    gray = image.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    grad = np.sqrt(gx**2 + gy**2)
    edge_density = float((grad > 0.1).mean())

    hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=False)
    probs = hist.astype(np.float64) / max(hist.sum(), 1)
    entropy = -float(np.sum([p * math.log2(p) for p in probs if p > 0]))

    return {
        "rgb_mean": image.mean(axis=(0, 1)).tolist(),
        "rgb_std": image.std(axis=(0, 1)).tolist(),
        "grayscale_mean": float(gray.mean()),
        "grayscale_std": float(gray.std()),
        "grayscale_entropy": entropy,
        "edge_density": edge_density,
    }


def aggregate_image_descriptors(descriptors: list[dict[str, float | list[float]]]) -> DatasetDescriptor:
    """Aggregate per-image descriptors into a dataset-level summary."""
    if not descriptors:
        raise ValueError("Cannot aggregate an empty descriptor list.")

    rgb_mean = np.asarray([item["rgb_mean"] for item in descriptors], dtype=np.float32)
    rgb_std = np.asarray([item["rgb_std"] for item in descriptors], dtype=np.float32)

    return DatasetDescriptor(
        num_samples=len(descriptors),
        rgb_mean=rgb_mean.mean(axis=0).astype(float).tolist(),
        rgb_std=rgb_std.mean(axis=0).astype(float).tolist(),
        grayscale_mean=float(np.mean([item["grayscale_mean"] for item in descriptors])),
        grayscale_std=float(np.mean([item["grayscale_std"] for item in descriptors])),
        grayscale_entropy=float(np.mean([item["grayscale_entropy"] for item in descriptors])),
        edge_density=float(np.mean([item["edge_density"] for item in descriptors])),
    )
