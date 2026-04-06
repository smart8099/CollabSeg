"""Checkpoint helpers shared across segmentation model loading paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Normalize common checkpoint payloads to a plain tensor state dict."""
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state", "model", "net"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload is not a recognized state dict format.")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        normalized = key[7:] if key.startswith("module.") else key
        state_dict[normalized] = value
    return state_dict


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str,
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> None:
    """Load a model checkpoint from either repo-native or raw upstream formats."""
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(payload)
    model.load_state_dict(state_dict, strict=strict)
