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


def select_prediction(predictions: list[PredictionRecord], config: dict) -> EnsembleDecision:
    """Choose the final ensemble output using consensus, margin, and fusion rules."""
    if not predictions:
        raise ValueError("No predictions provided.")

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

    ranking = [
        {
            "model_name": prediction.model_name,
            "score": prediction.score,
            "features": prediction.features,
        }
        for prediction in ranked
    ]
    return EnsembleDecision(
        decision_mode=decision_mode,
        selected_model=selected_model,
        final_mask=final_mask,
        final_probability_map=final_prob,
        ranking=ranking,
        reason=reason,
    )
