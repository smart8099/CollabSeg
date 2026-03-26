#!/usr/bin/env python3
"""Run agentic multi-model segmentation for a single input image and prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polypseg.ensemble import EnsembleOrchestrator, build_predictors, build_registry, load_registry_config, resolve_device


def main() -> None:
    """Load the ensemble, segment one image, and print the decision summary."""
    parser = argparse.ArgumentParser(description="Run agentic multi-model segmentation on one image.")
    parser.add_argument("--ensemble-config", type=str, default="configs/ensemble/default.yaml")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-mask", type=str, default="")
    args = parser.parse_args()

    device = resolve_device(args.device.lower())
    ensemble_config = load_registry_config(args.ensemble_config)
    registry = build_registry(args.ensemble_config, ROOT)
    predictors = build_predictors(registry, device=device)
    orchestrator = EnsembleOrchestrator(predictors=predictors, config=ensemble_config)

    image = Image.open(args.image).convert("RGB")
    result = orchestrator.run(image=image, prompt=args.prompt)
    decision = result["decision"]

    payload = {
        "decision_mode": decision.decision_mode,
        "selected_model": decision.selected_model,
        "reason": decision.reason,
        "ranking": decision.ranking,
    }
    print(json.dumps(payload, indent=2))

    if args.output_mask:
        output_path = Path(args.output_mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((decision.final_mask.astype(np.uint8) * 255)).save(output_path)


if __name__ == "__main__":
    main()
