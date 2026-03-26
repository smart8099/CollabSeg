"""Registry helpers for loading ensemble model specifications from config."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .types import ModelSpec


def load_registry_config(path: str | Path) -> dict:
    """Load the ensemble registry and scoring configuration from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_registry(config_path: str | Path, root_dir: str | Path) -> list[ModelSpec]:
    """Build model specifications from the registry config and checkpoint metadata."""
    root_dir = Path(root_dir)
    config = load_registry_config(config_path)
    model_specs: list[ModelSpec] = []
    for item in config.get("registry", {}).get("models", []):
        checkpoint_path = root_dir / item["checkpoint"]
        checkpoint_dir = checkpoint_path.parent
        merged_config_path = checkpoint_dir / "config_merged.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not merged_config_path.exists():
            raise FileNotFoundError(f"Missing config_merged.json for checkpoint: {checkpoint_path}")

        merged_config = json.loads(merged_config_path.read_text(encoding="utf-8"))
        model_cfg = merged_config["model"]
        data_cfg = merged_config["data"]
        model_specs.append(
            ModelSpec(
                name=item["name"],
                checkpoint=checkpoint_path,
                architecture=model_cfg["name"],
                model_params=model_cfg["params"],
                image_size=int(data_cfg["image_size"]),
                normalize_mean=list(data_cfg["normalize_mean"]),
                normalize_std=list(data_cfg["normalize_std"]),
                prompt_capable=bool(item.get("prompt_capable", False)),
                metadata={
                    "experiment_dir": str(checkpoint_dir),
                    "config_merged_path": str(merged_config_path),
                },
            )
        )
    return model_specs
