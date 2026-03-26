#!/usr/bin/env python3
"""Build a portable unified train/val/test split for the local polyp datasets."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
import shutil


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    """Return whether the given path is a supported image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith(".")


def source_from_train_name(name: str) -> str:
    """Infer the original source dataset from a pooled training filename."""
    stem = Path(name).stem
    return "CVC-ClinicDB" if stem.isdigit() else "Kvasir"


def build_records(dataset_root: Path) -> list[dict[str, str]]:
    """Collect image-mask records from the raw train and test dataset folders."""
    records: list[dict[str, str]] = []

    train_images = dataset_root / "TrainDataset" / "data" / "image"
    train_masks = dataset_root / "TrainDataset" / "data" / "masks"
    for image_path in sorted(p for p in train_images.iterdir() if is_image_file(p)):
        mask_path = train_masks / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path}")
        source_dataset = source_from_train_name(image_path.name)
        records.append(
            {
                "sample_id": f"{source_dataset}__{image_path.stem}",
                "source_dataset": source_dataset,
                "origin_split": "original_train",
                "raw_image_path": str(image_path.resolve()),
                "raw_mask_path": str(mask_path.resolve()),
            }
        )

    test_root = dataset_root / "TestDataset" / "data"
    for dataset_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        images_dir = dataset_dir / "images"
        masks_dir = dataset_dir / "masks"
        for image_path in sorted(p for p in images_dir.iterdir() if is_image_file(p)):
            mask_path = masks_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path}")
            records.append(
                {
                    "sample_id": f"{dataset_dir.name}__{image_path.stem}",
                    "source_dataset": dataset_dir.name,
                    "origin_split": "original_test",
                    "raw_image_path": str(image_path.resolve()),
                    "raw_mask_path": str(mask_path.resolve()),
                }
            )

    return records


def assign_splits(
    records: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    """Assign deterministic stratified train/val/test splits to manifest rows."""
    if round(train_ratio + val_ratio + test_ratio, 10) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_source[record["source_dataset"]].append(record)

    rng = random.Random(seed)
    assigned: list[dict[str, str]] = []
    for source, source_records in sorted(by_source.items()):
        shuffled = list(source_records)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count

        if total >= 3:
            if val_count == 0:
                val_count = 1
                train_count -= 1
            if test_count == 0:
                test_count = 1
                train_count -= 1

        boundaries = (
            ("train", train_count),
            ("val", val_count),
            ("test", test_count),
        )
        index = 0
        for split_name, count in boundaries:
            for record in shuffled[index : index + count]:
                record = dict(record)
                record["split"] = split_name
                assigned.append(record)
            index += count

    return sorted(assigned, key=lambda row: (row["split"], row["source_dataset"], row["sample_id"]))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a list of row dictionaries to a CSV file."""
    if not rows:
        raise ValueError("Cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def materialize_files(output_root: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Copy files into the split layout and replace raw paths with relative paths."""
    materialized: list[dict[str, str]] = []
    for row in rows:
        split_root = output_root / row["split"] / row["source_dataset"]
        image_dir = split_root / "images"
        mask_dir = split_root / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        image_target = image_dir / Path(row["raw_image_path"]).name
        mask_target = mask_dir / Path(row["raw_mask_path"]).name
        shutil.copy2(row["raw_image_path"], image_target)
        shutil.copy2(row["raw_mask_path"], mask_target)

        updated = dict(row)
        updated["image_path"] = str(image_target.relative_to(output_root.parent))
        updated["mask_path"] = str(mask_target.relative_to(output_root.parent))
        updated.pop("raw_image_path", None)
        updated.pop("raw_mask_path", None)
        materialized.append(updated)

    return sorted(materialized, key=lambda row: (row["split"], row["source_dataset"], row["sample_id"]))


def write_summary(path: Path, rows: list[dict[str, str]], seed: int) -> None:
    """Write a markdown summary of split counts and per-dataset composition."""
    split_counts = Counter(row["split"] for row in rows)
    per_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        per_dataset[row["source_dataset"]][row["split"]] += 1

    lines = [
        "Unified polyp dataset split summary",
        "",
        f"Seed: {seed}",
        f"Total samples: {len(rows)}",
        "",
        "Overall counts:",
        f"- train: {split_counts['train']}",
        f"- val: {split_counts['val']}",
        f"- test: {split_counts['test']}",
        "",
        "Per-dataset counts:",
    ]
    for dataset in sorted(per_dataset):
        counts = per_dataset[dataset]
        lines.append(
            f"- {dataset}: train={counts['train']}, val={counts['val']}, test={counts['test']}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Parse CLI arguments and generate the unified dataset split artifacts."""
    parser = argparse.ArgumentParser(description="Build a unified 80/10/10 split for local polyp datasets.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/agentpolyp_2504"),
        help="Root containing TrainDataset/ and TestDataset/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/agentpolyp_2504/unified_split"),
        help="Output directory for manifests and symlinked split folders.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = build_records(args.dataset_root)

    sample_ids = [row["sample_id"] for row in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id values detected.")

    manifests_dir = args.output_root / "manifests"
    files_dir = args.output_root / "files"
    summary_path = args.output_root / "SUMMARY.md"
    if manifests_dir.exists():
        shutil.rmtree(manifests_dir)
    if files_dir.exists():
        shutil.rmtree(files_dir)
    if summary_path.exists():
        summary_path.unlink()
    manifests_dir.mkdir(parents=True, exist_ok=True)

    split_rows = assign_splits(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    materialized_rows = materialize_files(files_dir, split_rows)
    manifest_rows = sorted(
        [{key: value for key, value in row.items() if key != "split"} for row in materialized_rows],
        key=lambda row: (row["source_dataset"], row["sample_id"]),
    )
    write_csv(manifests_dir / "all_samples.csv", manifest_rows)
    write_csv(manifests_dir / "all_splits.csv", materialized_rows)

    for split_name in ("train", "val", "test"):
        split_only = [row for row in materialized_rows if row["split"] == split_name]
        write_csv(manifests_dir / f"{split_name}.csv", split_only)

    write_summary(summary_path, materialized_rows, seed=args.seed)


if __name__ == "__main__":
    main()
