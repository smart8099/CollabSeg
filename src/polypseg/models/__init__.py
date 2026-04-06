"""Model registry and factory functions for segmentation architectures."""

from __future__ import annotations

import logging
from typing import Any

from .deeplabv3plus import DeepLabV3Plus
from .unet import UNet
from .unetpp import UNetPlusPlus
from .unetv2 import UNetV2

logger = logging.getLogger(__name__)

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
    model = MODEL_REGISTRY[key](**kwargs)
    _log_model_info(name, kwargs, model)
    return model


def _log_model_info(name: str, kwargs: dict[str, Any], model: Any) -> None:
    """Log a summary of the built model's architecture and parameter count."""
    lines = [f"[build_model] name={name}"]

    encoder_name = kwargs.get("encoder_name", "custom")
    encoder_pretrained = kwargs.get("encoder_pretrained", False)
    pretrained_path = kwargs.get("pretrained_path", "")
    checkpoint_path = kwargs.get("checkpoint_path", "")
    channel = kwargs.get("channel")
    features = kwargs.get("features")

    if name.lower() in {"unetv2", "deeplabv3plus", "deeplabv3+"}:
        if name.lower() == "unetv2":
            lines.append("  encoder      : pvt_v2_b2")
            lines.append(f"  channel      : {channel or 32}")
            lines.append(f"  deep sup     : {kwargs.get('deep_supervision', True)}")
            if pretrained_path:
                lines.append(f"  backbone ckpt: {pretrained_path}")
            if checkpoint_path:
                lines.append(f"  model ckpt   : {checkpoint_path}")
        else:
            lines.append(f"  encoder      : {encoder_name}")
            if encoder_name and encoder_name != "custom":
                lines.append(f"  pretrained   : {encoder_pretrained}")
        if features:
            lines.append(f"  features     : {list(features)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"  total params : {total_params:,}")
    lines.append(f"  trainable    : {trainable_params:,}")

    print("\n".join(lines), flush=True)


__all__ = [
    "DeepLabV3Plus",
    "MODEL_REGISTRY",
    "UNet",
    "UNetPlusPlus",
    "UNetV2",
    "build_model",
]
