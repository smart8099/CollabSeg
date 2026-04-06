#!/usr/bin/env python3
"""Train one segmentation model from the shared YAML configuration system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polypseg.data import PolypSegmentationDataset, build_eval_transforms, build_train_transforms
from polypseg.models import build_model
from polypseg.training import BCEDiceLoss, evaluate, load_config, prepare_output_dir, set_seed, train_one_epoch


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build the optimizer selected in the training config."""
    train_cfg = config["train"]
    lr = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    name = train_cfg["optimizer"].lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {train_cfg['optimizer']}")


def build_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    """Build the learning-rate scheduler selected in the training config."""
    train_cfg = config["train"]
    name = train_cfg["scheduler"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["epochs"],
            eta_min=train_cfg["min_learning_rate"],
        )
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {train_cfg['scheduler']}")


def resolve_device(requested: str) -> torch.device:
    """Resolve the preferred runtime device with CPU fallback."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    """Parse CLI arguments, train the model, and save training artifacts."""
    parser = argparse.ArgumentParser(description="Train a segmentation model from YAML config.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--smoke-run-batches", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config, args.model_config)
    model_name = str(config["model"]["name"])
    config["experiment_name"] = f"{config['experiment_name']}_{model_name}"
    if args.epochs > 0:
        config["train"]["epochs"] = args.epochs
    if args.smoke_run_batches >= 0:
        config["train"]["smoke_run_batches"] = args.smoke_run_batches
    set_seed(int(config["seed"]))

    output_dir = prepare_output_dir(config["output_dir"], config["experiment_name"])
    (output_dir / "config_merged.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    device = resolve_device(str(config["device"]).lower())
    data_cfg = config["data"]
    train_cfg = config["train"]
    loss_cfg = config["loss"]
    augment_cfg = data_cfg.get("augment", {})

    train_transform = build_train_transforms(
        image_size=int(data_cfg["image_size"]),
        mean=list(data_cfg["normalize_mean"]),
        std=list(data_cfg["normalize_std"]),
        horizontal_flip_prob=float(augment_cfg.get("horizontal_flip_prob", 0.5)),
        vertical_flip_prob=float(augment_cfg.get("vertical_flip_prob", 0.0)),
        rotate90_prob=float(augment_cfg.get("rotate90_prob", 0.5)),
        color_jitter_prob=float(augment_cfg.get("color_jitter_prob", 0.3)),
        brightness=float(augment_cfg.get("brightness", 0.15)),
        contrast=float(augment_cfg.get("contrast", 0.15)),
    )
    eval_transform = build_eval_transforms(
        image_size=int(data_cfg["image_size"]),
        mean=list(data_cfg["normalize_mean"]),
        std=list(data_cfg["normalize_std"]),
    )

    train_dataset = PolypSegmentationDataset(data_cfg["train_csv"], data_cfg["root_dir"], train_transform)
    val_dataset = PolypSegmentationDataset(data_cfg["val_csv"], data_cfg["root_dir"], eval_transform)
    test_dataset = PolypSegmentationDataset(data_cfg["test_csv"], data_cfg["root_dir"], eval_transform)

    loader_kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": bool(config["pin_memory"]),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = build_model(config["model"]["name"], **config["model"]["params"]).to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(loss_cfg["bce_weight"]),
        dice_weight=float(loss_cfg["dice_weight"]),
    )
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    use_amp = bool(train_cfg["mixed_precision"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    smoke_run_batches = int(train_cfg.get("smoke_run_batches", 0))

    best_val_dice = float("-inf")
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0))
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip_norm=float(train_cfg["grad_clip_norm"]),
            max_batches=smoke_run_batches,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=smoke_run_batches,
        )
        if scheduler is not None:
            scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics.get("dice", 0.0),
            "val_iou": val_metrics.get("iou", 0.0),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))

        if epoch_record["val_dice"] > best_val_dice:
            best_val_dice = float(epoch_record["val_dice"])
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "config": config}, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_val_dice": best_val_dice, "patience": early_stop_patience}))
            break

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=smoke_run_batches,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test": test_metrics}))


if __name__ == "__main__":
    main()
