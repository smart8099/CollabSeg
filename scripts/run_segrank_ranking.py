#!/usr/bin/env python3
"""Run dataset-level SegRank proposal and determination on a target manifest."""

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


def _load_rows(csv_path: Path, max_samples: int) -> list[dict[str, str]]:
    """Load manifest rows and optionally truncate them."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def main() -> None:
    """Parse arguments, run probe inference, and emit a SegRank ranking report."""
    parser = argparse.ArgumentParser(description="Run dataset-level SegRank ranking on a target manifest.")
    parser.add_argument("--ensemble-config", type=str, default="configs/ensemble/default.yaml")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/source_artifacts")
    parser.add_argument("--target-csv", type=str, required=True)
    parser.add_argument("--root-dir", type=str, default="datasets/agentpolyp_2504/unified_split")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--prescreen-top-k", type=int, default=0)
    parser.add_argument("--top-k-retrieval", type=int, default=3)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    from polypseg.ensemble import build_predictors, build_registry, load_registry_config, resolve_device
    from polypseg.segrank import (
        aggregate_evidence,
        aggregate_image_descriptors,
        aggregate_mask_morphology,
        compute_arbitration_alpha,
        compute_image_descriptor,
        compute_mask_morphology,
        compute_prediction_evidence,
        compute_prior_scores,
        determine_final_ranking,
        load_source_artifacts,
        retrieve_similar_datasets,
        score_model_compatibility,
        score_proposal_from_evidence,
        select_top_compatible_models,
        summarize_proposal_margin,
        write_json,
    )

    artifacts_summary = load_source_artifacts(ROOT / args.artifacts_dir)
    device = resolve_device(args.device.lower())
    ensemble_config_path = ROOT / args.ensemble_config
    ensemble_config = load_registry_config(ensemble_config_path)
    registry = build_registry(ensemble_config_path, ROOT)
    predictors = build_predictors(registry, device=device)
    threshold = float(ensemble_config["scoring"]["threshold"])

    rows = _load_rows(ROOT / args.target_csv, max_samples=int(args.max_samples))
    root_dir = ROOT / args.root_dir

    target_image_descriptors: list[dict[str, float | list[float]]] = []
    for row in rows:
        image = Image.open(root_dir / row["image_path"]).convert("RGB")
        image_np = np.asarray(image, dtype=np.uint8)
        target_image_descriptors.append(compute_image_descriptor(image_np))

    descriptor_summary = aggregate_image_descriptors(target_image_descriptors)
    compatibility_scores = score_model_compatibility(
        artifacts_summary=artifacts_summary,
        target_descriptor_embedding=descriptor_summary.embedding,
    )
    selected_model_names = select_top_compatible_models(
        compatibility_scores=compatibility_scores,
        top_k=int(args.prescreen_top_k),
    )
    selected_name_set = set(selected_model_names)
    selected_predictors = [predictor for predictor in predictors if predictor.spec.name in selected_name_set]
    if not selected_predictors:
        selected_predictors = predictors
        selected_model_names = [predictor.spec.name for predictor in predictors]

    target_morphology_features: list[dict[str, float]] = []
    evidence_by_model: dict[str, list[dict[str, float]]] = defaultdict(list)

    for row in rows:
        image = Image.open(root_dir / row["image_path"]).convert("RGB")
        image_np = np.asarray(image, dtype=np.uint8)

        predictions = [
            predictor.predict(image=image, threshold=threshold)
            for predictor in selected_predictors
        ]
        consensus_prob = np.mean([prediction.probability_map for prediction in predictions], axis=0)
        consensus_mask = (consensus_prob >= threshold).astype(np.uint8)
        target_morphology_features.append(compute_mask_morphology(consensus_mask))

        for prediction in predictions:
            evidence = compute_prediction_evidence(
                prediction=prediction,
                image_np=image_np,
                peer_predictions=predictions,
            )
            evidence_by_model[prediction.model_name].append(evidence)

    morphology_summary = aggregate_mask_morphology(target_morphology_features)
    retrieved = retrieve_similar_datasets(
        artifacts_summary=artifacts_summary,
        target_descriptor_embedding=descriptor_summary.embedding,
        target_morphology_embedding=morphology_summary.embedding,
        top_k=int(args.top_k_retrieval),
    )

    evidence_means_by_model: dict[str, dict[str, float]] = {}
    proposal_scores: dict[str, float] = {}
    for model_name, records in sorted(evidence_by_model.items()):
        summary = aggregate_evidence(records)
        evidence_means_by_model[model_name] = summary.feature_means
        proposal_scores[model_name] = score_proposal_from_evidence(summary.feature_means)

    prior_scores = compute_prior_scores(artifacts_summary=artifacts_summary, retrieved_datasets=retrieved)
    proposal_margin = summarize_proposal_margin(proposal_scores)
    alpha = compute_arbitration_alpha(proposal_margin=proposal_margin, retrieved_datasets=retrieved)
    ranking = determine_final_ranking(
        proposal_scores=proposal_scores,
        prior_scores=prior_scores,
        alpha=alpha,
        evidence_means_by_model=evidence_means_by_model,
    )

    payload = {
        "num_samples": len(rows),
        "target_descriptor": descriptor_summary.to_dict(),
        "compatibility_scores": compatibility_scores,
        "selected_models": selected_model_names,
        "target_morphology": morphology_summary.to_dict(),
        "retrieved_datasets": [item.to_dict() for item in retrieved],
        "proposal_margin": proposal_margin,
        "alpha": alpha,
        "ranking": [item.to_dict() for item in ranking],
    }

    print(json.dumps(payload, indent=2))
    if args.output_json:
        write_json(ROOT / args.output_json, payload)


if __name__ == "__main__":
    main()
