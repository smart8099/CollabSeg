#!/usr/bin/env python3
"""Evaluate the ensemble on a dataset using batched model inference in standalone evaluation space."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polypseg.ensemble import build_registry, dice_iou, load_registry_config, resolve_device
from polypseg.ensemble.policy import select_prediction
from polypseg.ensemble.scoring import score_prediction
from polypseg.ensemble.types import PredictionRecord
from polypseg.models import build_model
from polypseg.models.checkpointing import load_checkpoint_into_model


def _aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    """Average numeric fields across a list of per-sample metric records."""
    if not records:
        return {}
    result: dict[str, float] = {}
    numeric_keys = [key for key in records[0] if key not in {"sample_id", "source_dataset"}]
    for key in numeric_keys:
        result[key] = float(np.mean([record[key] for record in records]))
    return result


def _load_rgb(path: Path) -> Image.Image:
    """Load an RGB image from disk."""
    return Image.open(path).convert("RGB")


def _load_mask(path: Path) -> np.ndarray:
    """Load a binary mask from disk."""
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return (mask > 127).astype(np.uint8)


def _load_mask_resized(path: Path, image_size: int) -> np.ndarray:
    """Load a binary mask from disk and resize it into evaluation space."""
    mask = Image.open(path).convert("L").resize((image_size, image_size), Image.NEAREST)
    mask_np = np.asarray(mask, dtype=np.uint8)
    return (mask_np > 127).astype(np.uint8)


def _preprocess_batch(images: list[Image.Image], image_size: int, mean: list[float], std: list[float]) -> torch.Tensor:
    """Resize and normalize a list of PIL images into a batch tensor."""
    batch = []
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)
    for image in images:
        image = image.resize((image_size, image_size), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - mean_np) / std_np
        batch.append(image_np.transpose(2, 0, 1))
    return torch.tensor(batch, dtype=torch.float32)


@torch.no_grad()
def _predict_for_model(spec, rows: list[dict[str, str]], root_dir: Path, device: torch.device, batch_size: int, threshold: float):
    """Run one model over the full split in batches and return prediction records by sample id."""
    model = build_model(spec.architecture, **spec.model_params).to(device)
    load_checkpoint_into_model(model, spec.checkpoint, device=device)
    model.eval()

    results: dict[str, PredictionRecord] = {}
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_images = [_load_rgb(root_dir / row["image_path"]) for row in batch_rows]
        batch_tensor = _preprocess_batch(
            images=batch_images,
            image_size=spec.image_size,
            mean=spec.normalize_mean,
            std=spec.normalize_std,
        ).to(device)

        logits = model(batch_tensor)
        if isinstance(logits, list):
            logits = logits[-1]
        probabilities = torch.sigmoid(logits).detach().cpu()

        for row, prob_map in zip(batch_rows, probabilities):
            prob_np = np.asarray(prob_map[0].tolist(), dtype=np.float32)
            mask = (prob_np >= threshold).astype(np.uint8)
            confidence = float(prob_np[mask == 1].mean()) if mask.any() else float(prob_np.mean())
            results[row["sample_id"]] = PredictionRecord(
                model_name=spec.name,
                logits=prob_map.unsqueeze(0),
                probability_map=prob_np,
                mask=mask,
                confidence=confidence,
                metadata={"evaluation_size": spec.image_size},
            )
    return results


def main() -> None:
    """Parse CLI arguments and evaluate the ensemble with batched inference."""
    parser = argparse.ArgumentParser(description="Evaluate the ensemble selector with batched inference.")
    parser.add_argument("--ensemble-config", type=str, default="configs/ensemble/default.yaml")
    parser.add_argument("--split-csv", type=str, default="datasets/agentpolyp_2504/unified_split/manifests/test.csv")
    parser.add_argument("--root-dir", type=str, default="datasets/agentpolyp_2504/unified_split")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = resolve_device(args.device.lower())
    ensemble_config_path = ROOT / args.ensemble_config
    ensemble_config = load_registry_config(ensemble_config_path)
    registry = build_registry(ensemble_config_path, ROOT)
    threshold = float(ensemble_config["scoring"]["threshold"])

    root_dir = ROOT / args.root_dir
    split_csv = ROOT / args.split_csv
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    predictions_by_model = {
        spec.name: _predict_for_model(
            spec=spec,
            rows=rows,
            root_dir=root_dir,
            device=device,
            batch_size=int(args.batch_size),
            threshold=threshold,
        )
        for spec in registry
    }

    per_sample_records = []
    per_source = defaultdict(list)
    per_model_records = defaultdict(list)

    for row in rows:
        image_path = root_dir / row["image_path"]
        mask_path = root_dir / row["mask_path"]
        image = _load_rgb(image_path)
        image_for_features = image.resize((registry[0].image_size, registry[0].image_size), Image.BILINEAR)
        image_np = np.asarray(image_for_features)
        target_mask = _load_mask_resized(mask_path, registry[0].image_size)

        predictions = [predictions_by_model[spec.name][row["sample_id"]] for spec in registry]
        predictions = [
            score_prediction(
                prediction=prediction,
                prompt=args.prompt,
                image_np=image_np,
                peer_predictions=predictions,
                config=ensemble_config,
            )
            for prediction in predictions
        ]
        decision = select_prediction(predictions, ensemble_config)

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
        _, oracle_metrics = max(per_model_metrics.items(), key=lambda item: item[1]["dice"])
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
    best_standalone_model = max(standalone_summary.items(), key=lambda item: item[1]["dice"])[0]
    summary = {
        "num_samples": len(per_sample_records),
        "overall": _aggregate(per_sample_records),
        "per_dataset": {source: _aggregate(records) for source, records in sorted(per_source.items())},
        "standalone_models": standalone_summary,
        "best_standalone_model": best_standalone_model,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        output_path = ROOT / args.output_json if not Path(args.output_json).is_absolute() else Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
