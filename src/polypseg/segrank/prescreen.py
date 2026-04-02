"""Descriptor-based model pre-screening utilities for SegRank."""

from __future__ import annotations

from typing import Any

from .embeddings import cosine_similarity, embedding_distance


def score_model_compatibility(
    artifacts_summary: dict[str, Any],
    target_descriptor_embedding: list[float],
) -> dict[str, float]:
    """Score target compatibility for each model from descriptor-centroid similarity."""
    scores: dict[str, float] = {}
    for model_name, payload in artifacts_summary.get("operating_ranges", {}).items():
        centroid = payload.get("descriptor_centroid", {}).get("embedding")
        if centroid is None:
            scores[model_name] = 0.0
            continue
        similarity = cosine_similarity(target_descriptor_embedding, centroid)
        distance = embedding_distance(target_descriptor_embedding, centroid)
        scores[model_name] = float(similarity - 0.05 * distance)
    return scores


def select_top_compatible_models(
    compatibility_scores: dict[str, float],
    top_k: int,
) -> list[str]:
    """Select the top-K compatible model names."""
    ranked = sorted(compatibility_scores.items(), key=lambda item: item[1], reverse=True)
    if top_k <= 0:
        return [name for name, _ in ranked]
    return [name for name, _ in ranked[:top_k]]
