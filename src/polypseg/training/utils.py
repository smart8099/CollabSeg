"""Utility helpers shared across training scripts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_output_dir(root: str | Path, experiment_name: str) -> Path:
    """Create and return the output directory for one experiment run."""
    path = Path(root) / experiment_name
    path.mkdir(parents=True, exist_ok=True)
    return path
