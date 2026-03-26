#!/usr/bin/env python3
"""Evaluate the ensemble selector against standalone and oracle baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polypseg.ensemble import (
    EnsembleOrchestrator,
    build_predictors,
    build_registry,
    dice_iou,
    load_registry_config,
    resolve_device,
)


def _load_mask(mask_path: Path) -> np.ndarray:
    """Load a binary mask from disk and convert it to a uint8 array."""
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return (mask > 127).astype(np.uint8)


def _aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    """Average numeric fields across a list of per-sample metric records."""
    if not records:
        return {}
    result: dict[str, float] = {}
    numeric_keys = [key for key in records[0] if key not in {"sample_id", "source_dataset"}]
    for key in numeric_keys:
        result[key] = float(np.mean([record[key] for record in records]))
    return result


def main() -> None:
    """Parse CLI arguments and evaluate the ensemble over a manifest split."""
    parser = argparse.ArgumentParser(description="Evaluate the ensemble selector against standalone and oracle baselines.")
    parser.add_argument("--ensemble-config", type=str, default="configs/ensemble/default.yaml")
    parser.add_argument("--split-csv", type=str, default="datasets/agentpolyp_2504/unified_split/manifests/test.csv")
    parser.add_argument("--root-dir", type=str, default="datasets/agentpolyp_2504/unified_split")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device.lower())
    ensemble_config = load_registry_config(args.ensemble_config)
    registry = build_registry(args.ensemble_config, ROOT)
    predictors = build_predictors(registry, device=device)
    orchestrator = EnsembleOrchestrator(predictors=predictors, config=ensemble_config)

    root_dir = ROOT / args.root_dir
    split_csv = ROOT / args.split_csv
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    per_sample_records = []
    per_source = defaultdict(list)
    per_model_records = defaultdict(list)

    for row in rows:
        image_path = root_dir / row["image_path"]
        mask_path = root_dir / row["mask_path"]
        image = Image.open(image_path).convert("RGB")
        target_mask = _load_mask(mask_path)

        result = orchestrator.run(image=image, prompt=args.prompt)
        decision = result["decision"]
        predictions = result["predictions"]

        per_model_metrics = {
            pred.model_name: dice_iou(pred.mask, target_mask)
            for pred in predictions
        }
        for model_name, metrics in per_model_metrics.items():
            per_model_records[model_name].append(
                {
                    "sample_id": row["sample_id"],
                    "source_dataset": row["source_dataset"],
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                }
            )
        selector_metrics = dice_iou(decision.final_mask, target_mask)
        oracle_name, oracle_metrics = max(per_model_metrics.items(), key=lambda item: item[1]["dice"])

        selector_hit_oracle = float(selector_metrics["dice"] >= (oracle_metrics["dice"] - 1e-6))

        record = {
            "sample_id": row["sample_id"],
            "source_dataset": row["source_dataset"],
            "selector_dice": selector_metrics["dice"],
            "selector_iou": selector_metrics["iou"],
            "oracle_dice": oracle_metrics["dice"],
            "oracle_iou": oracle_metrics["iou"],
            "selector_hit_oracle": selector_hit_oracle,
            "selector_oracle_gap": oracle_metrics["dice"] - selector_metrics["dice"],
        }
        per_sample_records.append(record)
        per_source[row["source_dataset"]].append(record)

    standalone_summary = {
        model_name: _aggregate(records)
        for model_name, records in sorted(per_model_records.items())
    }
    best_standalone_model = max(
        standalone_summary.items(),
        key=lambda item: item[1]["dice"],
    )[0]
    summary = {
        "num_samples": len(per_sample_records),
        "overall": _aggregate(per_sample_records),
        "per_dataset": {source: _aggregate(records) for source, records in sorted(per_source.items())},
        "standalone_models": standalone_summary,
        "best_standalone_model": best_standalone_model,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
