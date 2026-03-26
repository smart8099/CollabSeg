"""Model registry and factory functions for segmentation architectures."""

from __future__ import annotations

from typing import Any

from .deeplabv3plus import DeepLabV3Plus
from .unet import UNet
from .unetpp import UNetPlusPlus
from .unetv2 import UNetV2


MODEL_REGISTRY = {
    "unet": UNet,
    "unet++": UNetPlusPlus,
    "unetpp": UNetPlusPlus,
    "unetv2": UNetV2,
    "deeplabv3+": DeepLabV3Plus,
    "deeplabv3plus": DeepLabV3Plus,
}


def build_model(name: str, **kwargs: Any):
    """Instantiate a segmentation model from the registry by name."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)


__all__ = [
    "DeepLabV3Plus",
    "MODEL_REGISTRY",
    "UNet",
    "UNetPlusPlus",
    "UNetV2",
    "build_model",
]
