"""Mask morphology feature extraction for SegRank."""

from __future__ import annotations

from collections import deque
import math

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


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    """Count 8-neighbors for each foreground pixel in a binary mask."""
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    counts = np.zeros_like(mask, dtype=np.uint8)
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for dy, dx in offsets:
        counts += padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return counts


def _transition_count(mask: np.ndarray) -> np.ndarray:
    """Count 0-to-1 neighbor transitions for Zhang-Suen thinning."""
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
    transitions = np.zeros_like(mask, dtype=np.uint8)
    for current, nxt in zip(neighbors[:-1], neighbors[1:]):
        transitions += ((current == 0) & (nxt == 1)).astype(np.uint8)
    return transitions


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Compute a thin binary skeleton using Zhang-Suen thinning."""
    skeleton = mask.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        neighbors = _neighbor_count(skeleton)
        transitions = _transition_count(skeleton)
        padded = np.pad(skeleton, 1, mode="constant")
        p2 = padded[:-2, 1:-1]
        p4 = padded[1:-1, 2:]
        p6 = padded[2:, 1:-1]
        p8 = padded[1:-1, :-2]

        remove = (
            (skeleton == 1)
            & (neighbors >= 2)
            & (neighbors <= 6)
            & (transitions == 1)
            & ((p2 * p4 * p6) == 0)
            & ((p4 * p6 * p8) == 0)
        )
        if np.any(remove):
            skeleton[remove] = 0
            changed = True

        neighbors = _neighbor_count(skeleton)
        transitions = _transition_count(skeleton)
        padded = np.pad(skeleton, 1, mode="constant")
        p2 = padded[:-2, 1:-1]
        p4 = padded[1:-1, 2:]
        p6 = padded[2:, 1:-1]
        p8 = padded[1:-1, :-2]

        remove = (
            (skeleton == 1)
            & (neighbors >= 2)
            & (neighbors <= 6)
            & (transitions == 1)
            & ((p2 * p4 * p8) == 0)
            & ((p2 * p6 * p8) == 0)
        )
        if np.any(remove):
            skeleton[remove] = 0
            changed = True
    return skeleton


def _component_sizes(mask: np.ndarray) -> list[int]:
    """Return the foreground sizes of 4-connected components."""
    binary = mask.astype(bool)
    visited = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    sizes: list[int] = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            size = 0
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            sizes.append(size)
    return sizes


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Compute the convex hull of 2D points with the monotonic chain algorithm."""
    if points.shape[0] <= 1:
        return points
    pts = sorted((float(x), float(y)) for y, x in points)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        """Return the 2D cross product for hull construction."""
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if not hull:
        return np.asarray(pts, dtype=np.float32)
    return np.asarray(hull, dtype=np.float32)


def _polygon_perimeter(points: np.ndarray) -> float:
    """Compute the perimeter of a polygon represented by ordered points."""
    if points.shape[0] <= 1:
        return 0.0
    shifted = np.roll(points, -1, axis=0)
    diff = shifted - points
    return float(np.sqrt(np.sum(diff * diff, axis=1)).sum())


def compute_mask_morphology(mask: np.ndarray) -> dict[str, float]:
    """Compute lightweight geometric properties for one binary mask."""
    binary = mask.astype(bool)
    height, width = binary.shape
    area_ratio = float(binary.mean())
    components = float(_component_count(binary))
    boundary = _binary_boundary(binary)
    boundary_ratio = float(boundary.mean())
    boundary_pixels = float(boundary.sum())

    if binary.any():
        ys, xs = np.where(binary)
        bbox_h = max(int(ys.max() - ys.min() + 1), 1)
        bbox_w = max(int(xs.max() - xs.min() + 1), 1)
        bbox_aspect_ratio = float(max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1))
        centroid_x = float(xs.mean() / max(width - 1, 1))
        centroid_y = float(ys.mean() / max(height - 1, 1))
        empty_mask = 0.0
        area_pixels = float(binary.sum())
        compactness = float((4.0 * math.pi * area_pixels) / max(boundary_pixels * boundary_pixels, 1.0))
        skeleton = _skeletonize(binary.astype(np.uint8))
        skeleton_neighbors = _neighbor_count(skeleton.astype(bool))
        branch_points = float(np.sum((skeleton == 1) & (skeleton_neighbors > 2)))
        skeleton_pixels = float(skeleton.sum())
        branching_index = branch_points / max(skeleton_pixels, 1.0)
        component_sizes = _component_sizes(binary)
        probs = np.asarray(component_sizes, dtype=np.float32) / max(float(sum(component_sizes)), 1.0)
        volume_distribution_entropy = float(-np.sum([p * math.log2(float(p)) for p in probs if p > 0]))
        hull = _convex_hull(np.argwhere(boundary > 0))
        hull_perimeter = _polygon_perimeter(hull)
        boundary_tortuosity = float(boundary_pixels / max(hull_perimeter, 1.0))
    else:
        bbox_aspect_ratio = 0.0
        centroid_x = 0.5
        centroid_y = 0.5
        empty_mask = 1.0
        compactness = 0.0
        branching_index = 0.0
        volume_distribution_entropy = 0.0
        boundary_tortuosity = 0.0

    features = {
        "area_ratio": area_ratio,
        "component_count": components,
        "compactness": compactness,
        "branching_index": branching_index,
        "volume_distribution_entropy": volume_distribution_entropy,
        "boundary_ratio": boundary_ratio,
        "boundary_tortuosity": boundary_tortuosity,
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
        mean_compactness=float(np.mean([item["compactness"] for item in features])),
        mean_branching_index=float(np.mean([item["branching_index"] for item in features])),
        mean_volume_distribution_entropy=float(np.mean([item["volume_distribution_entropy"] for item in features])),
        mean_boundary_ratio=float(np.mean([item["boundary_ratio"] for item in features])),
        mean_boundary_tortuosity=float(np.mean([item["boundary_tortuosity"] for item in features])),
        mean_bbox_aspect_ratio=float(np.mean([item["bbox_aspect_ratio"] for item in features])),
        std_bbox_aspect_ratio=float(np.std([item["bbox_aspect_ratio"] for item in features])),
        mean_centroid_x=float(np.mean([item["centroid_x"] for item in features])),
        mean_centroid_y=float(np.mean([item["centroid_y"] for item in features])),
        mean_empty_mask_ratio=float(np.mean([item["empty_mask"] for item in features])),
        embedding=embedding_summary["mean"],
    )
