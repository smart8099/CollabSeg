"""Top-level package exports for training, ensemble inference, and SegRank utilities."""

__all__: list[str] = []

try:
    from .ensemble import EnsembleOrchestrator, build_predictors, build_registry, resolve_device
    from .models import DeepLabV3Plus, UNet, UNetPlusPlus, UNetV2, build_model

    __all__ += [
        "UNet",
        "UNetPlusPlus",
        "UNetV2",
        "DeepLabV3Plus",
        "EnsembleOrchestrator",
        "build_model",
        "build_predictors",
        "build_registry",
        "resolve_device",
    ]
except ModuleNotFoundError:
    # Allow lightweight utilities to import even when optional runtime dependencies are absent.
    pass

from .segrank import (
    aggregate_evidence,
    aggregate_image_descriptors,
    aggregate_mask_morphology,
    compute_image_descriptor,
    compute_mask_morphology,
    compute_prediction_evidence,
)

__all__ += [
    "aggregate_evidence",
    "aggregate_image_descriptors",
    "aggregate_mask_morphology",
    "compute_image_descriptor",
    "compute_mask_morphology",
    "compute_prediction_evidence",
]
