"""SegRank source-side quality metrics and utility normalization helpers."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract a one-pixel approximate boundary from a binary mask."""
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


def _component_count(mask: np.ndarray) -> int:
    """Count 4-connected foreground components in a binary mask."""
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


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Return boundary coordinates for a binary mask as float32 points."""
    boundary = _binary_boundary(mask.astype(np.uint8))
    coords = np.argwhere(boundary > 0)
    if coords.size == 0 and mask.any():
        coords = np.argwhere(mask > 0)
    return coords.astype(np.float32)


def _min_distances(source_points: np.ndarray, target_points: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Compute per-point minimum Euclidean distances from source to target points."""
    if source_points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if target_points.size == 0:
        return np.full((source_points.shape[0],), np.inf, dtype=np.float32)

    mins = np.empty((source_points.shape[0],), dtype=np.float32)
    for start in range(0, source_points.shape[0], chunk_size):
        chunk = source_points[start : start + chunk_size]
        diff = chunk[:, None, :] - target_points[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        mins[start : start + chunk.shape[0]] = dist.min(axis=1)
    return mins


def topo_score(mask: np.ndarray, target: np.ndarray) -> float:
    """Compute a simple topology score from empty/non-empty and component agreement."""
    pred_non_empty = bool(mask.astype(bool).any())
    gt_non_empty = bool(target.astype(bool).any())
    empty_match = 1.0 if pred_non_empty == gt_non_empty else 0.0

    pred_components = float(_component_count(mask))
    gt_components = float(_component_count(target))
    component_score = 1.0 - abs(pred_components - gt_components) / (max(gt_components, 1.0) + 1.0)
    component_score = max(0.0, min(component_score, 1.0))
    return float(0.5 * empty_match + 0.5 * component_score)


def hd95_assd(mask: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute HD95 and ASSD between binary masks using boundary point distances."""
    pred = mask.astype(bool)
    gt = target.astype(bool)
    if not pred.any() and not gt.any():
        return {"hd95": 0.0, "assd": 0.0}

    diag = float(np.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2))
    if pred.any() != gt.any():
        return {"hd95": diag, "assd": diag}

    pred_points = _surface_points(pred)
    gt_points = _surface_points(gt)
    pred_to_gt = _min_distances(pred_points, gt_points)
    gt_to_pred = _min_distances(gt_points, pred_points)
    symmetric = np.concatenate([pred_to_gt, gt_to_pred], axis=0)

    hd95 = float(np.percentile(symmetric, 95)) if symmetric.size > 0 else 0.0
    assd = float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean())) if symmetric.size > 0 else 0.0
    return {"hd95": hd95, "assd": assd}


def composite_metrics(mask: np.ndarray, target: np.ndarray, overlap_metrics: dict[str, float]) -> dict[str, float]:
    """Combine overlap, boundary, and topology metrics for one prediction."""
    metrics = dict(overlap_metrics)
    metrics.update(hd95_assd(mask, target))
    metrics["topo_score"] = topo_score(mask, target)
    metrics["foreground_area_error"] = float(abs(mask.mean() - target.mean()))
    return metrics


def _robust_normalize(value: float, lower: float, upper: float, eps: float = 1e-6) -> float:
    """Normalize a scalar into [0, 1] using robust lower and upper bounds."""
    clipped = min(max(value, lower), upper)
    return float((clipped - lower) / (upper - lower + eps))


def compute_utility_statistics(
    model_dataset_payload: dict[str, dict[str, dict[str, Any]]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Compute robust normalization stats and composite utility for each source pair."""
    records: list[dict[str, Any]] = []
    for model_name, dataset_map in sorted(model_dataset_payload.items()):
        for source_dataset, payload in sorted(dataset_map.items()):
            metrics_mean = payload["metrics_mean"]
            records.append(
                {
                    "model_name": model_name,
                    "source_dataset": source_dataset,
                    "dice": float(metrics_mean["dice"]),
                    "hd95": float(metrics_mean["hd95"]),
                    "assd": float(metrics_mean["assd"]),
                    "topo_score": float(metrics_mean["topo_score"]),
                }
            )

    if not records:
        return {"weights": dict(weights), "normalization": {}}

    normalization: dict[str, dict[str, float]] = {}
    for metric_name in ("dice", "hd95", "assd", "topo_score"):
        values = np.asarray([record[metric_name] for record in records], dtype=np.float32)
        normalization[metric_name] = {
            "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }

    for record in records:
        dice_norm = _robust_normalize(record["dice"], normalization["dice"]["p5"], normalization["dice"]["p95"])
        hd95_norm = _robust_normalize(record["hd95"], normalization["hd95"]["p5"], normalization["hd95"]["p95"])
        assd_norm = _robust_normalize(record["assd"], normalization["assd"]["p5"], normalization["assd"]["p95"])
        topo_norm = _robust_normalize(
            record["topo_score"],
            normalization["topo_score"]["p5"],
            normalization["topo_score"]["p95"],
        )
        hd95_good = 1.0 - hd95_norm
        assd_good = 1.0 - assd_norm
        utility = (
            weights["dice_norm"] * dice_norm
            + weights["hd95_good"] * hd95_good
            + weights["assd_good"] * assd_good
            + weights["topo_norm"] * topo_norm
        )
        payload = model_dataset_payload[record["model_name"]][record["source_dataset"]]
        payload["metrics_normalized"] = {
            "dice_norm": float(dice_norm),
            "hd95_good": float(hd95_good),
            "assd_good": float(assd_good),
            "topo_norm": float(topo_norm),
        }
        payload["utility"] = float(utility)

    return {
        "weights": dict(weights),
        "normalization": normalization,
    }
