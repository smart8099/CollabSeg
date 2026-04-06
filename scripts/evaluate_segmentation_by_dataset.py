#!/usr/bin/env python3
"""Evaluate one segmentation checkpoint separately for each dataset inside a split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

from PIL import Image

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    DataLoader = None

    class Dataset:  # type: ignore[override]
        """Fallback dataset base so CLI help still works without torch installed."""

        pass

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FilteredPolypSegmentationDataset(Dataset):
    """Dataset view built from an in-memory subset of manifest rows."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        root_dir: str | Path,
        transform,
    ) -> None:
        """Store filtered rows, dataset root, and transform callable."""
        self.rows = rows
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of filtered rows."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """Load and return one filtered segmentation sample."""
        row = self.rows[index]
        image = Image.open(self.root_dir / row["image_path"]).convert("RGB")
        mask = Image.open(self.root_dir / row["mask_path"]).convert("L")
        image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": row["sample_id"],
            "source_dataset": row["source_dataset"],
        }


def resolve_device(requested: str) -> torch.device:
    """Resolve the preferred runtime device with CPU fallback."""
    if torch is None:
        raise ModuleNotFoundError("torch is required to evaluate segmentation checkpoints.")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Load manifest rows from a CSV file."""
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_rows_by_dataset(rows: list[dict[str, str]]) -> OrderedDict[str, list[dict[str, str]]]:
    """Group manifest rows by source dataset while preserving first-seen order."""
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["source_dataset"], []).append(row)
    return grouped


def main() -> None:
    """Parse arguments, evaluate a checkpoint per dataset, and print a JSON report."""
    parser = argparse.ArgumentParser(description="Evaluate one segmentation checkpoint per dataset within a split.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if torch is None or DataLoader is None:
        raise ModuleNotFoundError("torch is required to evaluate segmentation checkpoints.")

    from polypseg.data import build_eval_transforms
    from polypseg.models import build_model
    from polypseg.models.checkpointing import load_checkpoint_into_model
    from polypseg.training import BCEDiceLoss, evaluate, load_config

    config = load_config(args.config, args.model_config)
    device = resolve_device(str(config["device"]).lower())
    data_cfg = config["data"]
    loss_cfg = config["loss"]

    split_csv = Path(data_cfg[f"{args.split}_csv"])
    rows = load_rows(split_csv)
    grouped_rows = group_rows_by_dataset(rows)
    transform = build_eval_transforms(
        image_size=int(data_cfg["image_size"]),
        mean=list(data_cfg["normalize_mean"]),
        std=list(data_cfg["normalize_std"]),
    )

    model = build_model(config["model"]["name"], **config["model"]["params"]).to(device)
    load_checkpoint_into_model(model, args.checkpoint, device=device)
    criterion = BCEDiceLoss(
        bce_weight=float(loss_cfg["bce_weight"]),
        dice_weight=float(loss_cfg["dice_weight"]),
    )

    per_dataset: dict[str, dict[str, float | int]] = {}
    for source_dataset, dataset_rows in grouped_rows.items():
        dataset = FilteredPolypSegmentationDataset(
            rows=dataset_rows,
            root_dir=data_cfg["root_dir"],
            transform=transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=bool(config["pin_memory"]),
        )
        metrics = evaluate(model=model, loader=loader, criterion=criterion, device=device)
        per_dataset[source_dataset] = {
            "num_samples": len(dataset_rows),
            **metrics,
        }

    macro_metrics: dict[str, float] = {}
    if per_dataset:
        metric_keys = [key for key in next(iter(per_dataset.values())).keys() if key != "num_samples"]
        for key in metric_keys:
            macro_metrics[key] = sum(float(item[key]) for item in per_dataset.values()) / len(per_dataset)

    payload = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "model_config": args.model_config,
        "num_datasets": len(per_dataset),
        "num_samples": len(rows),
        "macro_average": macro_metrics,
        "per_dataset": per_dataset,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
