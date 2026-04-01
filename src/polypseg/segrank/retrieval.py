"""Source-dataset retrieval utilities for SegRank."""

from __future__ import annotations

from typing import Any

from .embeddings import cosine_similarity
from .types import RetrievedDataset


def retrieve_similar_datasets(
    artifacts_summary: dict[str, Any],
    target_descriptor_embedding: list[float],
    target_morphology_embedding: list[float],
    top_k: int = 3,
) -> list[RetrievedDataset]:
    """Retrieve the most similar source datasets for a target dataset."""
    retrieved: list[RetrievedDataset] = []
    for source_dataset, payload in artifacts_summary["datasets"].items():
        descriptor_embedding = payload["descriptor"]["embedding"]
        morphology_embedding = payload["morphology"]["embedding"]
        descriptor_similarity = cosine_similarity(target_descriptor_embedding, descriptor_embedding)
        morphology_similarity = cosine_similarity(target_morphology_embedding, morphology_embedding)
        combined_similarity = 0.5 * descriptor_similarity + 0.5 * morphology_similarity
        retrieved.append(
            RetrievedDataset(
                source_dataset=source_dataset,
                descriptor_similarity=descriptor_similarity,
                morphology_similarity=morphology_similarity,
                combined_similarity=combined_similarity,
            )
        )
    return sorted(retrieved, key=lambda item: item.combined_similarity, reverse=True)[:top_k]
