"""Dataclasses used by the ensemble inference layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ModelSpec:
    """Describe one registered segmentation model and how to load it."""

    name: str
    checkpoint: Path
    architecture: str
    model_params: dict[str, Any]
    image_size: int
    normalize_mean: list[float]
    normalize_std: list[float]
    prompt_capable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    """Store the prediction output and scoring metadata for one model."""

    model_name: str
    logits: torch.Tensor
    probability_map: np.ndarray
    mask: np.ndarray
    confidence: float
    features: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleDecision:
    """Capture the final ensemble decision and its ranking context."""

    decision_mode: str
    selected_model: str
    final_mask: np.ndarray
    final_probability_map: np.ndarray
    ranking: list[dict[str, Any]]
    reason: str
