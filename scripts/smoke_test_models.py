#!/usr/bin/env python3
"""Run quick forward-pass smoke tests for the available segmentation models."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polypseg.models import build_model


def main() -> None:
    """Instantiate each model and print the output tensor shapes."""
    x = torch.randn(2, 3, 256, 256)
    model_names = ["unet", "unet++", "unetv2", "deeplabv3+"]

    for name in model_names:
        model = build_model(name, in_channels=3, num_classes=1)
        model.eval()
        with torch.no_grad():
            y = model(x)
        if isinstance(y, list):
            shapes = [tuple(out.shape) for out in y]
            print(f"{name}: deep_supervision={shapes}")
        else:
            print(f"{name}: output_shape={tuple(y.shape)}")


if __name__ == "__main__":
    main()
