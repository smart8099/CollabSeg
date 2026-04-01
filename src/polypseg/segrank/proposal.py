"""Target-evidence-only proposal scoring for SegRank."""

from __future__ import annotations

from typing import Any


def score_proposal_from_evidence(evidence_summary: dict[str, float]) -> float:
    """Compute a proposal score using target evidence only."""
    agreement = float(evidence_summary["agreement_iou"])
    fg_conf = float(evidence_summary["foreground_confidence"])
    bg_conf = float(evidence_summary["background_confidence"])
    boundary = float(evidence_summary["boundary_strength"])
    entropy = float(evidence_summary["mean_entropy"])
    components = float(evidence_summary["component_count"])
    confidence_margin = float(evidence_summary["confidence_margin"])

    component_penalty = 1.0 / (1.0 + max(components - 1.0, 0.0))
    entropy_bonus = 1.0 - min(entropy, 1.0)
    boundary_bonus = min(boundary / 4.0, 1.0)

    return (
        0.25 * agreement
        + 0.20 * fg_conf
        + 0.15 * bg_conf
        + 0.15 * confidence_margin
        + 0.15 * entropy_bonus
        + 0.05 * component_penalty
        + 0.05 * boundary_bonus
    )


def summarize_proposal_margin(model_scores: dict[str, float]) -> float:
    """Compute the margin between the top two proposal scores."""
    if not model_scores:
        return 0.0
    ranked = sorted(model_scores.values(), reverse=True)
    if len(ranked) == 1:
        return float(ranked[0])
    return float(ranked[0] - ranked[1])
