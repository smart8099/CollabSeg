#!/usr/bin/env python3
"""Evaluate one trained segmentation model and optionally save predicted masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polypseg.data import PolypSegmentationDataset, build_eval_transforms
from polypseg.models import build_model
from polypseg.training import BCEDiceLoss, evaluate, load_config


def resolve_device(requested: str) -> torch.device:
    """Resolve the preferred runtime device with CPU fallback."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_predictions(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    root_dir: Path,
    output_dir: Path,
    image_size: int,
    mean: list[float],
    std: list[float],
    device: torch.device,
) -> None:
    """Run inference over manifest rows and save predicted masks to disk."""
    transform = build_eval_transforms(image_size=image_size, mean=mean, std=std)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        image_path = root_dir / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        mask_stub = Image.new("L", image.size, 0)
        image_tensor, _ = transform(image, mask_stub)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            if isinstance(logits, list):
                logits = logits[-1]
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred_mask = (probs >= 0.5).astype(np.uint8) * 255
        pred_image = Image.fromarray(pred_mask)
        pred_image = pred_image.resize(image.size, Image.NEAREST)
        source_dir = output_dir / row["source_dataset"]
        source_dir.mkdir(parents=True, exist_ok=True)
        pred_image.save(source_dir / f"{row['sample_id']}.png")


def main() -> None:
    """Parse CLI arguments, evaluate the checkpoint, and optionally export predictions."""
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--save-predictions-dir", type=str, default="")
    args = parser.parse_args()

    config = load_config(args.config, args.model_config)
    device = resolve_device(str(config["device"]).lower())
    data_cfg = config["data"]
    loss_cfg = config["loss"]

    split_csv = data_cfg[f"{args.split}_csv"]
    transform = build_eval_transforms(
        image_size=int(data_cfg["image_size"]),
        mean=list(data_cfg["normalize_mean"]),
        std=list(data_cfg["normalize_std"]),
    )
    dataset = PolypSegmentationDataset(split_csv, data_cfg["root_dir"], transform)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
    )

    model = build_model(config["model"]["name"], **config["model"]["params"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = BCEDiceLoss(
        bce_weight=float(loss_cfg["bce_weight"]),
        dice_weight=float(loss_cfg["dice_weight"]),
    )
    metrics = evaluate(model=model, loader=loader, criterion=criterion, device=device)
    print(json.dumps({args.split: metrics}, indent=2))

    if args.save_predictions_dir:
        with Path(split_csv).open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        save_predictions(
            model=model,
            rows=rows,
            root_dir=Path(data_cfg["root_dir"]),
            output_dir=Path(args.save_predictions_dir),
            image_size=int(data_cfg["image_size"]),
            mean=list(data_cfg["normalize_mean"]),
            std=list(data_cfg["normalize_std"]),
            device=device,
        )


if __name__ == "__main__":
    main()
