"""Orchestration layer that runs predictors, scores outputs, and makes a decision."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .policy import select_prediction
from .predictors import SegmentationPredictor
from .scoring import score_prediction


class EnsembleOrchestrator:
    """Coordinate multi-model inference, scoring, and final selection."""

    def __init__(self, predictors: list[SegmentationPredictor], config: dict) -> None:
        """Store the predictors and ensemble configuration."""
        self.predictors = predictors
        self.config = config

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
        decision = select_prediction(predictions, self.config)
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
