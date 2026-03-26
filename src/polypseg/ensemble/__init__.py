"""Public ensemble inference utilities exposed at the package level."""

from .orchestrator import EnsembleOrchestrator, resolve_device
from .policy import select_prediction
from .predictors import SegmentationPredictor, build_predictors
from .registry import build_registry, load_registry_config
from .scoring import dice_iou, pairwise_consensus_iou, score_prediction
from .types import EnsembleDecision, ModelSpec, PredictionRecord

__all__ = [
    "EnsembleDecision",
    "EnsembleOrchestrator",
    "ModelSpec",
    "PredictionRecord",
    "SegmentationPredictor",
    "build_predictors",
    "build_registry",
    "dice_iou",
    "load_registry_config",
    "pairwise_consensus_iou",
    "resolve_device",
    "score_prediction",
    "select_prediction",
]
