#!/usr/bin/env python3
"""Build offline SegRank source artifacts from the current supervised model bank."""

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


def _load_mask(mask_path: Path) -> np.ndarray:
    """Load a binary mask from disk."""
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return (mask > 127).astype(np.uint8)


def _aggregate_numeric(records: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """Compute mean and std for numeric record fields."""
    if not records:
        raise ValueError("Cannot aggregate an empty record list.")
    keys = sorted(records[0].keys())
    means = {key: float(np.mean([item[key] for item in records])) for key in keys}
    stds = {key: float(np.std([item[key] for item in records])) for key in keys}
    return means, stds


def main() -> None:
    """Parse CLI arguments and build offline source artifacts."""
    parser = argparse.ArgumentParser(description="Build source-side SegRank artifacts from a labeled split manifest.")
    parser.add_argument("--ensemble-config", type=str, default="configs/ensemble/default.yaml")
    parser.add_argument("--split-csv", type=str, default="datasets/agentpolyp_2504/unified_split/manifests/train.csv")
    parser.add_argument("--root-dir", type=str, default="datasets/agentpolyp_2504/unified_split")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/source_artifacts")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    from polypseg.ensemble import build_predictors, build_registry, dice_iou, load_registry_config, resolve_device
    from polypseg.segrank import (
        ModelDatasetArtifact,
        aggregate_evidence,
        aggregate_embedding_vectors,
        aggregate_image_descriptors,
        aggregate_mask_morphology,
        composite_metrics,
        compute_utility_statistics,
        compute_image_descriptor,
        compute_mask_morphology,
        compute_prediction_evidence,
        ensure_dir,
        write_json,
    )

    device = resolve_device(args.device.lower())
    ensemble_config_path = ROOT / args.ensemble_config
    ensemble_config = load_registry_config(ensemble_config_path)
    registry = build_registry(ensemble_config_path, ROOT)
    predictors = build_predictors(registry, device=device)
    threshold = float(ensemble_config["scoring"]["threshold"])

    root_dir = ROOT / args.root_dir
    split_csv = ROOT / args.split_csv
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    utility_weights = {
        "dice_norm": 0.40,
        "hd95_good": 0.20,
        "assd_good": 0.20,
        "topo_norm": 0.20,
    }

    artifacts_dir = ensure_dir(ROOT / args.artifacts_dir)
    datasets_dir = ensure_dir(artifacts_dir / "datasets")
    embeddings_dir = ensure_dir(artifacts_dir / "embeddings")
    dataset_embeddings_dir = ensure_dir(embeddings_dir / "datasets")
    image_embeddings_dir = ensure_dir(embeddings_dir / "images")
    models_dir = ensure_dir(artifacts_dir / "models")

    per_dataset_image_descriptors: dict[str, list[dict[str, float | list[float]]]] = defaultdict(list)
    per_dataset_morphology: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_model_dataset_metrics: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    per_model_dataset_evidence: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    per_dataset_image_embeddings: dict[str, list[list[float]]] = defaultdict(list)
    per_dataset_morph_embeddings: dict[str, list[list[float]]] = defaultdict(list)
    per_model_dataset_response_embeddings: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    per_image_embedding_records: dict[str, list[dict[str, object]]] = defaultdict(list)

    for index, row in enumerate(rows, start=1):
        image_path = root_dir / row["image_path"]
        mask_path = root_dir / row["mask_path"]
        source_dataset = row["source_dataset"]

        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image, dtype=np.uint8)
        target_mask = _load_mask(mask_path)

        image_descriptor = compute_image_descriptor(image_np)
        morphology_features = compute_mask_morphology(target_mask)
        per_dataset_image_descriptors[source_dataset].append(image_descriptor)
        per_dataset_morphology[source_dataset].append(morphology_features)
        per_dataset_image_embeddings[source_dataset].append(list(image_descriptor["embedding"]))
        per_dataset_morph_embeddings[source_dataset].append(list(morphology_features["embedding"]))

        predictions = [
            predictor.predict(image=image, threshold=threshold)
            for predictor in predictors
        ]
        image_record = {
            "sample_id": row["sample_id"],
            "image_embedding": image_descriptor["embedding"],
            "morphology_embedding": morphology_features["embedding"],
            "model_response_embeddings": {},
        }
        for prediction in predictions:
            evidence = compute_prediction_evidence(
                prediction=prediction,
                image_np=image_np,
                peer_predictions=predictions,
            )
            metrics = composite_metrics(prediction.mask, target_mask, dice_iou(prediction.mask, target_mask))
            key = (prediction.model_name, source_dataset)
            per_model_dataset_metrics[key].append(metrics)
            per_model_dataset_evidence[key].append(evidence)
            per_model_dataset_response_embeddings[key].append(list(evidence["embedding"]))
            image_record["model_response_embeddings"][prediction.model_name] = evidence["embedding"]

        per_image_embedding_records[source_dataset].append(image_record)

        if index % 25 == 0:
            print(json.dumps({"processed_samples": index, "total_samples": len(rows)}))

    dataset_payload: dict[str, dict[str, object]] = {}
    for source_dataset in sorted(per_dataset_image_descriptors):
        dataset_payload[source_dataset] = {
            "descriptor": aggregate_image_descriptors(per_dataset_image_descriptors[source_dataset]).to_dict(),
            "morphology": aggregate_mask_morphology(per_dataset_morphology[source_dataset]).to_dict(),
        }
        write_json(datasets_dir / f"{source_dataset}.json", dataset_payload[source_dataset])
        write_json(
            dataset_embeddings_dir / f"{source_dataset}.json",
            {
                "image_embedding_summary": aggregate_embedding_vectors(per_dataset_image_embeddings[source_dataset]),
                "morphology_embedding_summary": aggregate_embedding_vectors(per_dataset_morph_embeddings[source_dataset]),
            },
        )
        write_json(
            image_embeddings_dir / f"{source_dataset}.json",
            {
                "source_dataset": source_dataset,
                "records": per_image_embedding_records[source_dataset],
            },
        )

    model_dataset_payload: dict[str, dict[str, object]] = defaultdict(dict)
    for model_name, source_dataset in sorted(per_model_dataset_metrics):
        metrics_mean, metrics_std = _aggregate_numeric(per_model_dataset_metrics[(model_name, source_dataset)])
        artifact = ModelDatasetArtifact(
            model_name=model_name,
            source_dataset=source_dataset,
            num_samples=len(per_model_dataset_metrics[(model_name, source_dataset)]),
            metrics_mean=metrics_mean,
            metrics_std=metrics_std,
            evidence=aggregate_evidence(per_model_dataset_evidence[(model_name, source_dataset)]),
        )
        model_dataset_payload[model_name][source_dataset] = artifact.to_dict()
        model_dataset_payload[model_name][source_dataset]["response_embedding_summary"] = aggregate_embedding_vectors(
            per_model_dataset_response_embeddings[(model_name, source_dataset)]
        )

    utility_stats = compute_utility_statistics(
        model_dataset_payload=model_dataset_payload,
        weights=utility_weights,
    )

    operating_ranges: dict[str, dict[str, object]] = {}
    for model_name, dataset_map in sorted(model_dataset_payload.items()):
        scores = {
            dataset_name: float(payload.get("utility", payload["metrics_mean"]["dice"]))
            for dataset_name, payload in dataset_map.items()
        }
        if not scores:
            continue
        dice_values = np.asarray(list(scores.values()), dtype=np.float32)
        threshold_score = float(np.median(dice_values))
        selected = [name for name, score in scores.items() if score >= threshold_score]
        selected = selected or list(scores.keys())
        centroid_rgb_mean = np.asarray(
            [dataset_payload[name]["descriptor"]["rgb_mean"] for name in selected],
            dtype=np.float32,
        ).mean(axis=0)
        centroid_edge_density = float(
            np.mean([dataset_payload[name]["descriptor"]["edge_density"] for name in selected])
        )
        operating_ranges[model_name] = {
            "selected_datasets": selected,
            "dice_threshold": threshold_score,
            "descriptor_centroid": {
                "rgb_mean": centroid_rgb_mean.astype(float).tolist(),
                "edge_density": centroid_edge_density,
                "embedding": np.asarray(
                    [dataset_payload[name]["descriptor"]["embedding"] for name in selected],
                    dtype=np.float32,
                ).mean(axis=0).astype(float).tolist(),
            },
        }
        write_json(models_dir / f"{model_name}.json", model_dataset_payload[model_name])

    summary = {
        "num_samples": len(rows),
        "num_datasets": len(dataset_payload),
        "num_models": len(model_dataset_payload),
        "datasets": dataset_payload,
        "models": model_dataset_payload,
        "operating_ranges": operating_ranges,
        "utility_statistics": utility_stats,
    }
    write_json(artifacts_dir / "summary.json", summary)
    print(json.dumps({"artifacts_dir": str(artifacts_dir), "num_samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
