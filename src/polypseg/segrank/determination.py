"""Prior-aware determination scoring and arbitration for SegRank."""

from __future__ import annotations

from typing import Any

from .types import ModelRankingRecord, RetrievedDataset


def compute_prior_scores(
    artifacts_summary: dict[str, Any],
    retrieved_datasets: list[RetrievedDataset],
) -> dict[str, float]:
    """Compute source-prior scores for each model from retrieved datasets."""
    prior_scores: dict[str, float] = {}
    for model_name, dataset_map in artifacts_summary["models"].items():
        weighted_sum = 0.0
        weight_total = 0.0
        for retrieved in retrieved_datasets:
            if retrieved.source_dataset not in dataset_map:
                continue
            similarity = max(retrieved.combined_similarity, 0.0)
            dice = float(dataset_map[retrieved.source_dataset]["metrics_mean"]["dice"])
            weighted_sum += similarity * dice
            weight_total += similarity
        prior_scores[model_name] = weighted_sum / weight_total if weight_total > 0 else 0.0
    return prior_scores


def compute_arbitration_alpha(proposal_margin: float, retrieved_datasets: list[RetrievedDataset]) -> float:
    """Compute a heuristic proposal-versus-prior blend weight."""
    proposal_term = max(0.0, min(proposal_margin / 0.15, 1.0))
    shift = 1.0 - max((retrieved_datasets[0].combined_similarity if retrieved_datasets else 0.0), 0.0)
    shift_term = max(0.0, min(shift, 1.0))
    alpha = 0.35 + 0.4 * proposal_term + 0.25 * shift_term
    return max(0.1, min(alpha, 0.9))


def determine_final_ranking(
    proposal_scores: dict[str, float],
    prior_scores: dict[str, float],
    alpha: float,
    evidence_means_by_model: dict[str, dict[str, float]],
) -> list[ModelRankingRecord]:
    """Blend proposal and prior scores into the final determined ranking."""
    records: list[ModelRankingRecord] = []
    for model_name in sorted(proposal_scores):
        proposal_score = float(proposal_scores[model_name])
        prior_score = float(prior_scores.get(model_name, 0.0))
        final_score = alpha * proposal_score + (1.0 - alpha) * prior_score
        records.append(
            ModelRankingRecord(
                model_name=model_name,
                proposal_score=proposal_score,
                prior_score=prior_score,
                final_score=final_score,
                evidence_summary=evidence_means_by_model[model_name],
            )
        )
    return sorted(records, key=lambda item: item.final_score, reverse=True)
