"""Predictor wrappers that turn trained models into a common inference interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from polypseg.models import build_model

from .types import ModelSpec, PredictionRecord


def _image_to_tensor(image: Image.Image, image_size: int, mean: list[float], std: list[float]) -> torch.Tensor:
    """Resize and normalize a PIL image into a batched tensor."""
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = (image_np - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return torch.tensor(image_np.transpose(2, 0, 1).tolist(), dtype=torch.float32).unsqueeze(0)


class SegmentationPredictor:
    """Load one trained segmentation model and run standardized inference."""

    def __init__(self, spec: ModelSpec, device: torch.device) -> None:
        """Load the model checkpoint and prepare the predictor for inference."""
        self.spec = spec
        self.device = device
        self.model = build_model(spec.architecture, **spec.model_params).to(device)
        checkpoint = torch.load(spec.checkpoint, map_location=device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image, prompt: str = "", threshold: float = 0.5) -> PredictionRecord:
        """Run inference on one image and return a normalized prediction record."""
        original_size = image.size
        image_tensor = _image_to_tensor(
            image=image,
            image_size=self.spec.image_size,
            mean=self.spec.normalize_mean,
            std=self.spec.normalize_std,
        ).to(self.device)

        logits = self.model(image_tensor)
        if isinstance(logits, list):
            logits = logits[-1]

        prob_map = np.asarray(torch.sigmoid(logits)[0, 0].detach().cpu().tolist(), dtype=np.float32)
        prob_image = Image.fromarray((prob_map * 255).astype(np.uint8)).resize(original_size, Image.BILINEAR)
        prob_map_resized = np.asarray(prob_image, dtype=np.float32) / 255.0
        mask = (prob_map_resized >= threshold).astype(np.uint8)
        confidence = float(prob_map_resized[mask == 1].mean()) if mask.any() else float(prob_map_resized.mean())

        return PredictionRecord(
            model_name=self.spec.name,
            logits=logits.detach().cpu(),
            probability_map=prob_map_resized,
            mask=mask,
            confidence=confidence,
            metadata={
                "prompt_used": bool(prompt and self.spec.prompt_capable),
                "original_size": list(original_size),
            },
        )


def build_predictors(specs: list[ModelSpec], device: torch.device) -> list[SegmentationPredictor]:
    """Instantiate predictor wrappers for all registered model specs."""
    return [SegmentationPredictor(spec=spec, device=device) for spec in specs]
