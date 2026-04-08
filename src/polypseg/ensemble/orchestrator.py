"""Orchestration layer that runs predictors, scores outputs, and makes a decision."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from polypseg.segrank import (
    compute_image_descriptor,
    compute_mask_morphology,
    compute_prior_scores,
    load_source_artifacts,
    retrieve_similar_datasets,
    score_model_compatibility,
)

from .policy import select_prediction
from .predictors import SegmentationPredictor
from .scoring import score_prediction


class EnsembleOrchestrator:
    """Coordinate multi-model inference, scoring, and final selection."""

    def __init__(self, predictors: list[SegmentationPredictor], config: dict) -> None:
        """Store the predictors and ensemble configuration."""
        self.predictors = predictors
        self.config = config
        self._source_artifacts = None
        artifacts_dir = (
            self.config.get("scoring", {})
            .get("policy", {})
            .get("anchor", {})
            .get("source_priors", {})
            .get("artifacts_dir", "")
        )
        if artifacts_dir:
            self._source_artifacts = load_source_artifacts(artifacts_dir)

    def _build_prior_context(
        self,
        image_np: np.ndarray,
        predictions,
    ) -> dict:
        """Build per-image source-prior context for prior-aware anchor decisions."""
        if self._source_artifacts is None:
            return {}

        anchor_cfg = self.config.get("scoring", {}).get("policy", {}).get("anchor", {})
        anchor_name = str(anchor_cfg.get("model_name", "")).strip()
        if not anchor_name:
            return {}

        anchor_prediction = next((prediction for prediction in predictions if prediction.model_name == anchor_name), None)
        if anchor_prediction is None:
            return {}

        descriptor = compute_image_descriptor(image_np)
        compatibility_scores = score_model_compatibility(
            artifacts_summary=self._source_artifacts,
            target_descriptor_embedding=descriptor["embedding"],
            distance_penalty=float(anchor_cfg.get("source_priors", {}).get("distance_penalty", 0.05)),
        )
        anchor_morphology = compute_mask_morphology(anchor_prediction.mask)
        retrieved = retrieve_similar_datasets(
            artifacts_summary=self._source_artifacts,
            target_descriptor_embedding=descriptor["embedding"],
            target_morphology_embedding=anchor_morphology["embedding"],
            top_k=int(anchor_cfg.get("source_priors", {}).get("top_k_retrieval", 3)),
        )
        prior_scores = compute_prior_scores(
            artifacts_summary=self._source_artifacts,
            retrieved_datasets=retrieved,
        )
        return {
            "prior_scores": prior_scores,
            "compatibility_scores": compatibility_scores,
            "retrieved_datasets": [item.to_dict() for item in retrieved],
            "anchor_descriptor_embedding": descriptor["embedding"],
            "anchor_morphology_embedding": anchor_morphology["embedding"],
        }

    def run(self, image: Image.Image, prompt: str = "") -> dict:
        """Run all predictors on an image and return the ranked ensemble decision."""
        image_np = np.asarray(image.convert("RGB"))
        threshold = float(self.config["scoring"]["threshold"])

        predictions = [predictor.predict(image=image, prompt=prompt, threshold=threshold) for predictor in self.predictors]
        predictions = [
            score_prediction(
                prediction=prediction,
                prompt=prompt,
                image_np=image_np,
                peer_predictions=predictions,
                config=self.config,
            )
            for prediction in predictions
        ]
        decision = select_prediction(
            predictions,
            self.config,
            prior_context=self._build_prior_context(image_np=image_np, predictions=predictions),
        )
        return {
            "prompt": prompt,
            "predictions": predictions,
            "decision": decision,
        }


def resolve_device(requested: str) -> torch.device:
    """Resolve the preferred runtime device with CPU fallback."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
