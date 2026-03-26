"""Configuration loading and merging helpers for training scripts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into a base configuration mapping."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and ensure it parses to a mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def load_config(base_config_path: str | Path, model_config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the base config and optionally merge in a model-specific config."""
    config = load_yaml(base_config_path)
    if model_config_path is not None:
        model_cfg = load_yaml(model_config_path)
        config = _deep_merge(config, model_cfg)
    return config
