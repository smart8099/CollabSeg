"""Mask morphology feature extraction for SegRank."""

from __future__ import annotations

from collections import deque

import numpy as np

from .embeddings import aggregate_embedding_vectors, morphology_embedding_from_features
from .types import MorphologyDescriptor


def _component_count(mask: np.ndarray) -> int:
    """Count 4-connected foreground components in a binary mask."""
    binary = mask.astype(bool)
    if not binary.any():
        return 0

    visited = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    count = 0
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


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract a one-pixel-wide approximate mask boundary."""
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


def compute_mask_morphology(mask: np.ndarray) -> dict[str, float]:
    """Compute lightweight geometric properties for one binary mask."""
    binary = mask.astype(bool)
    height, width = binary.shape
    area_ratio = float(binary.mean())
    components = float(_component_count(binary))
    boundary_ratio = float(_binary_boundary(binary).mean())

    if binary.any():
        ys, xs = np.where(binary)
        bbox_h = max(int(ys.max() - ys.min() + 1), 1)
        bbox_w = max(int(xs.max() - xs.min() + 1), 1)
        bbox_aspect_ratio = float(max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1))
        centroid_x = float(xs.mean() / max(width - 1, 1))
        centroid_y = float(ys.mean() / max(height - 1, 1))
        empty_mask = 0.0
    else:
        bbox_aspect_ratio = 0.0
        centroid_x = 0.5
        centroid_y = 0.5
        empty_mask = 1.0

    features = {
        "area_ratio": area_ratio,
        "component_count": components,
        "boundary_ratio": boundary_ratio,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "empty_mask": empty_mask,
    }
    features["embedding"] = morphology_embedding_from_features(features)
    return features


def aggregate_mask_morphology(features: list[dict[str, float]]) -> MorphologyDescriptor:
    """Aggregate per-mask morphology features into a dataset summary."""
    if not features:
        raise ValueError("Cannot aggregate an empty morphology feature list.")

    embedding_summary = aggregate_embedding_vectors([item["embedding"] for item in features])

    return MorphologyDescriptor(
        num_samples=len(features),
        mean_area_ratio=float(np.mean([item["area_ratio"] for item in features])),
        std_area_ratio=float(np.std([item["area_ratio"] for item in features])),
        mean_component_count=float(np.mean([item["component_count"] for item in features])),
        mean_boundary_ratio=float(np.mean([item["boundary_ratio"] for item in features])),
        mean_bbox_aspect_ratio=float(np.mean([item["bbox_aspect_ratio"] for item in features])),
        mean_centroid_x=float(np.mean([item["centroid_x"] for item in features])),
        mean_centroid_y=float(np.mean([item["centroid_y"] for item in features])),
        mean_empty_mask_ratio=float(np.mean([item["empty_mask"] for item in features])),
        embedding=embedding_summary["mean"],
    )
