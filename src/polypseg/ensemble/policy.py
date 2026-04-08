"""Decision rules for selecting or fusing ensemble predictions."""

from __future__ import annotations

import numpy as np

from .scoring import pairwise_consensus_iou
from .types import EnsembleDecision, PredictionRecord


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the IoU between two binary masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _fuse_predictions(predictions: list[PredictionRecord], top_k: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Average top prediction probability maps and threshold the result."""
    selected = predictions[:top_k]
    prob_stack = np.stack([prediction.probability_map for prediction in selected], axis=0)
    fused_prob = prob_stack.mean(axis=0)
    fused_mask = (fused_prob >= threshold).astype(np.uint8)
    return fused_mask, fused_prob


def _trust_score(prediction: PredictionRecord, weights: dict[str, float]) -> float:
    """Compute a model-trust score from already-scored heuristic features."""
    features = prediction.features
    return float(
        float(weights.get("confidence", 0.0)) * float(features.get("confidence_score", 0.0))
        + float(weights.get("agreement", 0.0)) * float(features.get("agreement_score", 0.0))
        + float(weights.get("shape", 0.0)) * float(features.get("shape_score", 0.0))
        + float(weights.get("boundary", 0.0)) * float(features.get("boundary_score", 0.0))
    )


def _ranking_payload(
    ranked: list[PredictionRecord],
    extra_fields_by_model: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    """Build the ranking payload returned in ensemble decisions."""
    extra_fields_by_model = extra_fields_by_model or {}
    return [
        {
            "model_name": prediction.model_name,
            "score": prediction.score,
            "features": {
                **prediction.features,
                **extra_fields_by_model.get(prediction.model_name, {}),
            },
        }
        for prediction in ranked
    ]


def _select_prediction_legacy(predictions: list[PredictionRecord], config: dict) -> EnsembleDecision:
    """Choose the final ensemble output using the original consensus and margin rules."""
    ranked = sorted(predictions, key=lambda record: record.score, reverse=True)
    policy_cfg = config["scoring"]["policy"]
    threshold = float(config["scoring"]["threshold"])
    consensus_iou = pairwise_consensus_iou(ranked)

    if consensus_iou >= float(policy_cfg["consensus_iou_threshold"]):
        final_mask, final_prob = _fuse_predictions(ranked, top_k=len(ranked), threshold=threshold)
        reason = f"High consensus across models (mean pairwise IoU={consensus_iou:.3f})."
        decision_mode = "consensus"
        selected_model = "consensus"
    else:
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        margin = top.score - second.score if second is not None else 1.0
        overlap = _mask_iou(top.mask, second.mask) if second is not None else 1.0

        if second is not None and margin < float(policy_cfg["top_score_margin"]) and overlap >= float(
            policy_cfg["fuse_iou_threshold"]
        ):
            final_mask, final_prob = _fuse_predictions(
                ranked,
                top_k=int(policy_cfg["fuse_top_k"]),
                threshold=threshold,
            )
            reason = f"Top models were close in score (margin={margin:.3f}) with strong overlap (IoU={overlap:.3f})."
            decision_mode = "fuse_top_k"
            selected_model = "+".join(pred.model_name for pred in ranked[: int(policy_cfg["fuse_top_k"])])
        else:
            final_mask = top.mask
            final_prob = top.probability_map
            reason = f"Selected highest-scoring model with score margin={margin:.3f}."
            decision_mode = "select_best"
            selected_model = top.model_name

    return EnsembleDecision(
        decision_mode=decision_mode,
        selected_model=selected_model,
        final_mask=final_mask,
        final_probability_map=final_prob,
        ranking=_ranking_payload(ranked),
        reason=reason,
    )


def _select_prediction_anchor_override(predictions: list[PredictionRecord], config: dict) -> EnsembleDecision:
    """Select conservatively by anchoring on one strong model and only overriding with strong evidence."""
    ranked = sorted(predictions, key=lambda record: record.score, reverse=True)
    policy_cfg = config["scoring"]["policy"]
    threshold = float(config["scoring"]["threshold"])
    anchor_cfg = policy_cfg.get("anchor", {})
    anchor_name = str(anchor_cfg.get("model_name", "")).strip()
    if not anchor_name:
        return _select_prediction_legacy(predictions, config)

    anchor = next((prediction for prediction in ranked if prediction.model_name == anchor_name), None)
    if anchor is None:
        return _select_prediction_legacy(predictions, config)

    trust_weights = anchor_cfg.get(
        "trust_weights",
        {"confidence": 0.35, "agreement": 0.20, "shape": 0.30, "boundary": 0.15},
    )
    anchor_trust = _trust_score(anchor, trust_weights)
    extra_fields_by_model = {
        prediction.model_name: {"trust_score": _trust_score(prediction, trust_weights)}
        for prediction in ranked
    }

    anchor_trust_threshold = float(anchor_cfg.get("trust_threshold", 0.72))
    if anchor_trust >= anchor_trust_threshold:
        return EnsembleDecision(
            decision_mode="anchor_keep",
            selected_model=anchor.model_name,
            final_mask=anchor.mask,
            final_probability_map=anchor.probability_map,
            ranking=_ranking_payload(ranked, extra_fields_by_model=extra_fields_by_model),
            reason=f"Kept anchor model {anchor.model_name} because trust={anchor_trust:.3f} exceeded the stay threshold.",
        )

    alternatives = [prediction for prediction in ranked if prediction.model_name != anchor.model_name]
    if not alternatives:
        return EnsembleDecision(
            decision_mode="anchor_keep",
            selected_model=anchor.model_name,
            final_mask=anchor.mask,
            final_probability_map=anchor.probability_map,
            ranking=_ranking_payload(ranked, extra_fields_by_model=extra_fields_by_model),
            reason=f"Kept anchor model {anchor.model_name} because no alternative models were available.",
        )

    best_alt = alternatives[0]
    best_alt_trust = extra_fields_by_model[best_alt.model_name]["trust_score"]
    score_gain = float(best_alt.score - anchor.score)
    trust_gain = float(best_alt_trust - anchor_trust)
    anchor_overlap = _mask_iou(anchor.mask, best_alt.mask)
    disagreement = 1.0 - anchor_overlap

    override_score_margin = float(anchor_cfg.get("override_score_margin", 0.08))
    override_trust_margin = float(anchor_cfg.get("override_trust_margin", 0.02))
    override_min_disagreement = float(anchor_cfg.get("override_min_disagreement", 0.10))
    override_min_alt_trust = float(anchor_cfg.get("override_min_alt_trust", 0.68))

    allow_fusion = bool(anchor_cfg.get("allow_fusion", False))
    fusion_score_margin = float(anchor_cfg.get("fusion_score_margin", 0.03))
    fusion_min_anchor_trust = float(anchor_cfg.get("fusion_min_anchor_trust", 0.70))
    fusion_min_alt_trust = float(anchor_cfg.get("fusion_min_alt_trust", 0.72))
    fusion_iou_threshold = float(anchor_cfg.get("fusion_iou_threshold", 0.80))

    if (
        score_gain >= override_score_margin
        and trust_gain >= override_trust_margin
        and disagreement >= override_min_disagreement
        and best_alt_trust >= override_min_alt_trust
    ):
        if (
            allow_fusion
            and score_gain <= fusion_score_margin
            and anchor_overlap >= fusion_iou_threshold
            and anchor_trust >= fusion_min_anchor_trust
            and best_alt_trust >= fusion_min_alt_trust
        ):
            final_mask, final_prob = _fuse_predictions([best_alt, anchor], top_k=2, threshold=threshold)
            return EnsembleDecision(
                decision_mode="anchor_fuse",
                selected_model=f"{anchor.model_name}+{best_alt.model_name}",
                final_mask=final_mask,
                final_probability_map=final_prob,
                ranking=_ranking_payload(ranked, extra_fields_by_model=extra_fields_by_model),
                reason=(
                    f"Overrode anchor {anchor.model_name} with conservative fusion because alternative {best_alt.model_name} "
                    f"was stronger (score_gain={score_gain:.3f}, trust_gain={trust_gain:.3f}) while remaining highly consistent."
                ),
            )

        return EnsembleDecision(
            decision_mode="anchor_override",
            selected_model=best_alt.model_name,
            final_mask=best_alt.mask,
            final_probability_map=best_alt.probability_map,
            ranking=_ranking_payload(ranked, extra_fields_by_model=extra_fields_by_model),
            reason=(
                f"Overrode anchor {anchor.model_name} with {best_alt.model_name} because it showed a strong relative advantage "
                f"(score_gain={score_gain:.3f}, trust_gain={trust_gain:.3f}, disagreement={disagreement:.3f})."
            ),
        )

    return EnsembleDecision(
        decision_mode="anchor_keep",
        selected_model=anchor.model_name,
        final_mask=anchor.mask,
        final_probability_map=anchor.probability_map,
        ranking=_ranking_payload(ranked, extra_fields_by_model=extra_fields_by_model),
        reason=(
            f"Kept anchor {anchor.model_name}; best alternative {best_alt.model_name} did not clear the override criteria "
            f"(score_gain={score_gain:.3f}, trust_gain={trust_gain:.3f}, disagreement={disagreement:.3f})."
        ),
    )


def select_prediction(predictions: list[PredictionRecord], config: dict) -> EnsembleDecision:
    """Choose the final ensemble output using the configured selection policy."""
    if not predictions:
        raise ValueError("No predictions provided.")
    policy_cfg = config["scoring"]["policy"]
    selector_mode = str(policy_cfg.get("selector_mode", "legacy")).lower()
    if selector_mode == "anchor_override":
        return _select_prediction_anchor_override(predictions, config)
    return _select_prediction_legacy(predictions, config)
