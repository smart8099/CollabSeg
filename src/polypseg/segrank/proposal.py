"""Target-evidence-only proposal scoring for SegRank."""

from __future__ import annotations

from typing import Any


def score_proposal_from_evidence(
    evidence_summary: dict[str, float],
    weights: dict[str, float] | None = None,
    modifiers: dict[str, float] | None = None,
) -> float:
    """Compute a proposal score using target evidence only."""
    weights = weights or {}
    modifiers = modifiers or {}

    agreement = float(evidence_summary["agreement_iou"])
    fg_conf = float(evidence_summary["foreground_confidence"])
    bg_conf = float(evidence_summary["background_confidence"])
    boundary = float(evidence_summary["boundary_strength"])
    entropy = float(evidence_summary["mean_entropy"])
    components = float(evidence_summary["component_count"])
    confidence_margin = float(evidence_summary["confidence_margin"])

    component_offset = float(modifiers.get("component_offset", 1.0))
    component_penalty = 1.0 / (1.0 + max(components - component_offset, 0.0))
    entropy_cap = float(modifiers.get("entropy_cap", 1.0))
    entropy_bonus = 1.0 - min(entropy, entropy_cap)
    boundary_scale = max(float(modifiers.get("boundary_scale", 4.0)), 1e-6)
    boundary_bonus = min(boundary / boundary_scale, 1.0)

    return (
        float(weights.get("agreement", 0.25)) * agreement
        + float(weights.get("foreground_confidence", 0.20)) * fg_conf
        + float(weights.get("background_confidence", 0.15)) * bg_conf
        + float(weights.get("confidence_margin", 0.15)) * confidence_margin
        + float(weights.get("entropy_bonus", 0.15)) * entropy_bonus
        + float(weights.get("component_penalty", 0.05)) * component_penalty
        + float(weights.get("boundary_bonus", 0.05)) * boundary_bonus
    )


def summarize_proposal_margin(model_scores: dict[str, float]) -> float:
    """Compute the margin between the top two proposal scores."""
    if not model_scores:
        return 0.0
    ranked = sorted(model_scores.values(), reverse=True)
    if len(ranked) == 1:
        return float(ranked[0])
    return float(ranked[0] - ranked[1])
