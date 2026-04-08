"""Validation-time tuning for ensemble and SegRank selector heuristics."""

from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from polypseg.ensemble import PredictionRecord, build_registry, dice_iou, load_registry_config, resolve_device, score_prediction
from polypseg.ensemble.policy import select_prediction
from polypseg.models import build_model
from polypseg.models.checkpointing import load_checkpoint_into_model
from polypseg.segrank import (
    aggregate_evidence,
    aggregate_image_descriptors,
    aggregate_mask_morphology,
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
from polypseg.segrank.determination import compute_arbitration_alpha_with_params


@dataclass
class CachedSample:
    """Store one validation sample in a shared evaluation space."""

    sample_id: str
    source_dataset: str
    image_np: np.ndarray
    target_mask: np.ndarray
    probability_maps: dict[str, np.ndarray]


def load_tuning_config(path: str | Path) -> dict[str, Any]:
    """Load a tuning YAML file into a mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Tuning config at {path} must be a mapping.")
    return payload


def _aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    """Average numeric fields across records."""
    if not records:
        return {}
    keys = [key for key in records[0] if key not in {"sample_id", "source_dataset"}]
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def _set_nested(mapping: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path inside a nested mapping."""
    current = mapping
    parts = path.split(".")
    for key in parts[:-1]:
        current = current.setdefault(key, {})
    current[parts[-1]] = value


def _get_nested(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted path from a nested mapping."""
    current: Any = mapping
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _normalize_groups(config: dict[str, Any], groups: list[list[str]]) -> None:
    """Normalize path groups so their values sum to one."""
    for paths in groups:
        values = [float(_get_nested(config, path, 0.0)) for path in paths]
        total = sum(values)
        if total <= 0:
            continue
        for path, value in zip(paths, values):
            _set_nested(config, path, float(value / total))


def _load_rows(csv_path: Path, max_samples: int) -> list[dict[str, str]]:
    """Load manifest rows and optionally truncate them."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _load_rgb(path: Path) -> Image.Image:
    """Load an RGB image from disk."""
    return Image.open(path).convert("RGB")


def _load_mask_resized(path: Path, image_size: int) -> np.ndarray:
    """Load a binary mask and resize it into the shared evaluation space."""
    mask = Image.open(path).convert("L").resize((image_size, image_size), Image.NEAREST)
    mask_np = np.asarray(mask, dtype=np.uint8)
    return (mask_np > 127).astype(np.uint8)


def _preprocess_batch(images: list[Image.Image], image_size: int, mean: list[float], std: list[float]) -> torch.Tensor:
    """Resize and normalize a list of PIL images into a batch tensor."""
    mean_np = np.asarray(mean, dtype=np.float32)
    std_np = np.asarray(std, dtype=np.float32)
    batch = []
    for image in images:
        resized = image.resize((image_size, image_size), Image.BILINEAR)
        image_np = np.asarray(resized, dtype=np.float32) / 255.0
        image_np = (image_np - mean_np) / std_np
        batch.append(image_np.transpose(2, 0, 1))
    return torch.tensor(np.asarray(batch, dtype=np.float32), dtype=torch.float32)


@torch.no_grad()
def _predict_for_model(
    spec,
    rows: list[dict[str, str]],
    root_dir: Path,
    device: torch.device,
    batch_size: int,
    common_image_size: int,
) -> dict[str, np.ndarray]:
    """Run one model over a split and cache resized probability maps by sample id."""
    model = build_model(spec.architecture, **spec.model_params).to(device)
    load_checkpoint_into_model(model, spec.checkpoint, device=device)
    model.eval()

    results: dict[str, np.ndarray] = {}
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
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()

        for row, prob_map in zip(batch_rows, probabilities):
            prob_uint8 = (np.clip(prob_map[0], 0.0, 1.0) * 255.0).astype(np.uint8)
            resized = Image.fromarray(prob_uint8).resize((common_image_size, common_image_size), Image.BILINEAR)
            results[row["sample_id"]] = np.asarray(resized, dtype=np.float32) / 255.0

    return results


def _build_cache(
    root_dir: Path,
    rows: list[dict[str, str]],
    registry: list[Any],
    device: torch.device,
    batch_size: int,
) -> tuple[list[CachedSample], int]:
    """Cache images, masks, and per-model probability maps in one evaluation space."""
    image_sizes = {int(spec.image_size) for spec in registry}
    if len(image_sizes) != 1:
        raise ValueError(f"Selector tuning currently requires one shared image size, got {sorted(image_sizes)}")
    common_image_size = next(iter(image_sizes))

    predictions_by_model = {
        spec.name: _predict_for_model(
            spec=spec,
            rows=rows,
            root_dir=root_dir,
            device=device,
            batch_size=batch_size,
            common_image_size=common_image_size,
        )
        for spec in registry
    }

    cache: list[CachedSample] = []
    for row in rows:
        image = _load_rgb(root_dir / row["image_path"]).resize((common_image_size, common_image_size), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.uint8)
        target_mask = _load_mask_resized(root_dir / row["mask_path"], common_image_size)
        probability_maps = {spec.name: predictions_by_model[spec.name][row["sample_id"]] for spec in registry}
        cache.append(
            CachedSample(
                sample_id=row["sample_id"],
                source_dataset=row["source_dataset"],
                image_np=image_np,
                target_mask=target_mask,
                probability_maps=probability_maps,
            )
        )

    return cache, common_image_size


def _build_prediction_records(
    sample: CachedSample,
    model_names: list[str],
    threshold: float,
) -> list[PredictionRecord]:
    """Create fresh prediction records for one cached sample."""
    predictions: list[PredictionRecord] = []
    for model_name in model_names:
        prob_map = sample.probability_maps[model_name]
        mask = (prob_map >= threshold).astype(np.uint8)
        confidence = float(prob_map[mask == 1].mean()) if mask.any() else float(prob_map.mean())
        predictions.append(
            PredictionRecord(
                model_name=model_name,
                logits=torch.empty(0),
                probability_map=prob_map.copy(),
                mask=mask,
                confidence=confidence,
            )
        )
    return predictions


def _sample_parameter(name: str, spec: dict[str, Any], rng: random.Random) -> Any:
    """Sample one parameter according to the supplied search spec."""
    if "values" in spec:
        values = list(spec["values"])
        if not values:
            raise ValueError(f"Parameter '{name}' has an empty values list.")
        return rng.choice(values)
    if "uniform" in spec:
        low, high = spec["uniform"]
        return float(rng.uniform(float(low), float(high)))
    if "int_uniform" in spec:
        low, high = spec["int_uniform"]
        return int(rng.randint(int(low), int(high)))
    if "fixed" in spec:
        return spec["fixed"]
    raise ValueError(f"Parameter '{name}' must define one of: values, uniform, int_uniform, fixed.")


def _resolve_trial_payload(
    base_config: dict[str, Any],
    tuning_config: dict[str, Any],
    sampled_parameters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply sampled parameters onto the base config and selector-runtime config."""
    config = deepcopy(base_config)
    selector_runtime = deepcopy(tuning_config.get("selector_runtime", {}))

    for param_name, value in sampled_parameters.items():
        spec = tuning_config["search"]["parameters"][param_name]
        path = str(spec["path"])
        if path.startswith("segrank."):
            _set_nested(selector_runtime, path[len("segrank.") :], value)
        else:
            _set_nested(config, path, value)

    _normalize_groups(config, list(tuning_config.get("normalize_groups", [])))
    return config, selector_runtime


def _summarize_selected_model(
    selected_model: str,
    per_model_records: dict[str, list[dict[str, float]]],
) -> dict[str, float]:
    """Return aggregate metrics for the chosen model."""
    return _aggregate(per_model_records.get(selected_model, []))


def _evaluate_ensemble_trial(
    cache: list[CachedSample],
    model_names: list[str],
    trial_config: dict[str, Any],
    prompt: str,
    artifacts_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one ensemble selector configuration against cached validation data."""
    threshold = float(trial_config["scoring"]["threshold"])
    per_sample_records: list[dict[str, float | str]] = []
    per_source: dict[str, list[dict[str, float]]] = {}
    per_model_records: dict[str, list[dict[str, float]]] = {name: [] for name in model_names}
    anchor_cfg = trial_config.get("scoring", {}).get("policy", {}).get("anchor", {})

    for sample in cache:
        predictions = _build_prediction_records(sample, model_names=model_names, threshold=threshold)
        predictions = [
            score_prediction(
                prediction=prediction,
                prompt=prompt,
                image_np=sample.image_np,
                peer_predictions=predictions,
                config=trial_config,
            )
            for prediction in predictions
        ]
        prior_context = {}
        if artifacts_summary is not None and anchor_cfg.get("source_priors", {}).get("enabled", False):
            descriptor = compute_image_descriptor(sample.image_np)
            compatibility_scores = score_model_compatibility(
                artifacts_summary=artifacts_summary,
                target_descriptor_embedding=descriptor["embedding"],
                distance_penalty=float(anchor_cfg.get("source_priors", {}).get("distance_penalty", 0.05)),
            )
            anchor_name = str(anchor_cfg.get("model_name", "")).strip()
            anchor_prediction = next((prediction for prediction in predictions if prediction.model_name == anchor_name), None)
            if anchor_prediction is not None:
                anchor_morphology = compute_mask_morphology(anchor_prediction.mask)
                retrieved = retrieve_similar_datasets(
                    artifacts_summary=artifacts_summary,
                    target_descriptor_embedding=descriptor["embedding"],
                    target_morphology_embedding=anchor_morphology["embedding"],
                    top_k=int(anchor_cfg.get("source_priors", {}).get("top_k_retrieval", 3)),
                )
                prior_context = {
                    "prior_scores": compute_prior_scores(
                        artifacts_summary=artifacts_summary,
                        retrieved_datasets=retrieved,
                    ),
                    "compatibility_scores": compatibility_scores,
                    "retrieved_datasets": [item.to_dict() for item in retrieved],
                }

        decision = select_prediction(predictions, trial_config, prior_context=prior_context)

        per_model_metrics = {prediction.model_name: dice_iou(prediction.mask, sample.target_mask) for prediction in predictions}
        for model_name, metrics in per_model_metrics.items():
            per_model_records[model_name].append(
                {
                    "sample_id": sample.sample_id,
                    "source_dataset": sample.source_dataset,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                }
            )

        selector_metrics = dice_iou(decision.final_mask, sample.target_mask)
        oracle_name, oracle_metrics = max(per_model_metrics.items(), key=lambda item: item[1]["dice"])
        record = {
            "sample_id": sample.sample_id,
            "source_dataset": sample.source_dataset,
            "selector_dice": selector_metrics["dice"],
            "selector_iou": selector_metrics["iou"],
            "oracle_dice": oracle_metrics["dice"],
            "oracle_iou": oracle_metrics["iou"],
            "selector_hit_oracle": float(selector_metrics["dice"] >= (oracle_metrics["dice"] - 1e-6)),
            "selector_oracle_gap": oracle_metrics["dice"] - selector_metrics["dice"],
        }
        per_sample_records.append(record)
        per_source.setdefault(sample.source_dataset, []).append(record)

    standalone_summary = {model_name: _aggregate(records) for model_name, records in sorted(per_model_records.items())}
    best_standalone_model = max(standalone_summary.items(), key=lambda item: item[1].get("dice", 0.0))[0]
    overall = _aggregate(per_sample_records)
    overall["best_standalone_dice"] = standalone_summary[best_standalone_model]["dice"]
    overall["selector_vs_best_standalone_gap"] = overall["selector_dice"] - standalone_summary[best_standalone_model]["dice"]

    return {
        "mode": "ensemble",
        "num_samples": len(cache),
        "overall": overall,
        "per_dataset": {source: _aggregate(records) for source, records in sorted(per_source.items())},
        "standalone_models": standalone_summary,
        "best_standalone_model": best_standalone_model,
    }


def _evaluate_segrank_trial(
    cache: list[CachedSample],
    model_names: list[str],
    trial_config: dict[str, Any],
    selector_runtime: dict[str, Any],
    artifacts_summary: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one SegRank selector configuration against cached validation data."""
    threshold = float(trial_config["scoring"]["threshold"])
    compatibility_cfg = selector_runtime.get("compatibility", {})
    proposal_cfg = selector_runtime.get("proposal", {})
    arbitration_cfg = selector_runtime.get("arbitration", {})
    retrieval_cfg = selector_runtime.get("retrieval", {})
    prescreen_top_k = int(selector_runtime.get("prescreen_top_k", 0))
    top_k_retrieval = int(retrieval_cfg.get("top_k", selector_runtime.get("top_k_retrieval", 3)))

    target_descriptors = [compute_image_descriptor(sample.image_np) for sample in cache]
    descriptor_summary = aggregate_image_descriptors(target_descriptors)
    compatibility_scores = score_model_compatibility(
        artifacts_summary=artifacts_summary,
        target_descriptor_embedding=descriptor_summary.embedding,
        distance_penalty=float(compatibility_cfg.get("distance_penalty", 0.05)),
    )
    selected_model_names = select_top_compatible_models(compatibility_scores=compatibility_scores, top_k=prescreen_top_k)
    if not selected_model_names:
        selected_model_names = list(model_names)

    evidence_by_model: dict[str, list[dict[str, float]]] = {name: [] for name in selected_model_names}
    target_morphology_features: list[dict[str, float]] = []
    per_model_records: dict[str, list[dict[str, float]]] = {name: [] for name in selected_model_names}
    proposal_hit_oracle_flags: list[float] = []
    oracle_models: list[str] = []

    proposal_weights = proposal_cfg.get("weights")
    proposal_modifiers = proposal_cfg.get("modifiers")

    for sample in cache:
        predictions = _build_prediction_records(sample, model_names=selected_model_names, threshold=threshold)
        consensus_prob = np.mean([prediction.probability_map for prediction in predictions], axis=0)
        consensus_mask = (consensus_prob >= threshold).astype(np.uint8)
        target_morphology_features.append(compute_mask_morphology(consensus_mask))

        sample_model_scores: dict[str, float] = {}
        sample_model_dice: dict[str, float] = {}
        for prediction in predictions:
            evidence = compute_prediction_evidence(
                prediction=prediction,
                image_np=sample.image_np,
                peer_predictions=predictions,
            )
            evidence_by_model[prediction.model_name].append(evidence)
            sample_model_scores[prediction.model_name] = score_proposal_from_evidence(
                evidence_summary=evidence,
                weights=proposal_weights,
                modifiers=proposal_modifiers,
            )
            metrics = dice_iou(prediction.mask, sample.target_mask)
            sample_model_dice[prediction.model_name] = metrics["dice"]
            per_model_records[prediction.model_name].append(
                {
                    "sample_id": sample.sample_id,
                    "source_dataset": sample.source_dataset,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                }
            )

        oracle_model = max(sample_model_dice.items(), key=lambda item: item[1])[0]
        proposal_top_model = max(sample_model_scores.items(), key=lambda item: item[1])[0]
        proposal_hit_oracle_flags.append(float(proposal_top_model == oracle_model))
        oracle_models.append(oracle_model)

    morphology_summary = aggregate_mask_morphology(target_morphology_features)
    retrieved = retrieve_similar_datasets(
        artifacts_summary=artifacts_summary,
        target_descriptor_embedding=descriptor_summary.embedding,
        target_morphology_embedding=morphology_summary.embedding,
        top_k=top_k_retrieval,
    )

    evidence_means_by_model: dict[str, dict[str, float]] = {}
    proposal_scores: dict[str, float] = {}
    for model_name, records in sorted(evidence_by_model.items()):
        summary = aggregate_evidence(records)
        evidence_means_by_model[model_name] = summary.feature_means
        proposal_scores[model_name] = score_proposal_from_evidence(
            evidence_summary=summary.feature_means,
            weights=proposal_weights,
            modifiers=proposal_modifiers,
        )

    prior_scores = compute_prior_scores(artifacts_summary=artifacts_summary, retrieved_datasets=retrieved)
    proposal_margin = summarize_proposal_margin(proposal_scores)
    alpha = compute_arbitration_alpha_with_params(
        proposal_margin=proposal_margin,
        retrieved_datasets=retrieved,
        params=arbitration_cfg,
    )
    ranking = determine_final_ranking(
        proposal_scores=proposal_scores,
        prior_scores=prior_scores,
        alpha=alpha,
        evidence_means_by_model=evidence_means_by_model,
    )

    full_top_model = ranking[0].model_name
    prior_top_model = max(prior_scores.items(), key=lambda item: item[1])[0]
    proposal_top_model = max(proposal_scores.items(), key=lambda item: item[1])[0]
    standalone_summary = {model_name: _aggregate(records) for model_name, records in sorted(per_model_records.items())}
    best_standalone_model = max(standalone_summary.items(), key=lambda item: item[1].get("dice", 0.0))[0]
    selected_summary = _summarize_selected_model(full_top_model, per_model_records)

    overall = {
        "full_top_model_mean_dice": float(selected_summary.get("dice", 0.0)),
        "full_top_model_mean_iou": float(selected_summary.get("iou", 0.0)),
        "proposal_hit_oracle_rate": float(np.mean(proposal_hit_oracle_flags)) if proposal_hit_oracle_flags else 0.0,
        "full_hit_oracle_rate": float(np.mean([float(full_top_model == oracle_model) for oracle_model in oracle_models])) if oracle_models else 0.0,
        "prior_hit_oracle_rate": float(np.mean([float(prior_top_model == oracle_model) for oracle_model in oracle_models])) if oracle_models else 0.0,
        "best_standalone_dice": float(standalone_summary[best_standalone_model]["dice"]),
        "full_vs_best_standalone_gap": float(selected_summary.get("dice", 0.0) - standalone_summary[best_standalone_model]["dice"]),
    }

    return {
        "mode": "segrank",
        "num_samples": len(cache),
        "overall": overall,
        "selected_models": selected_model_names,
        "compatibility_scores": compatibility_scores,
        "retrieved_datasets": [item.to_dict() for item in retrieved],
        "proposal_margin": proposal_margin,
        "alpha": alpha,
        "proposal_top_model": proposal_top_model,
        "prior_top_model": prior_top_model,
        "full_top_model": full_top_model,
        "standalone_models": standalone_summary,
        "best_standalone_model": best_standalone_model,
        "ranking": [item.to_dict() for item in ranking],
    }


def _evaluate_trial(
    mode: str,
    cache: list[CachedSample],
    model_names: list[str],
    trial_config: dict[str, Any],
    selector_runtime: dict[str, Any],
    prompt: str,
    artifacts_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate one trial for the requested selector mode."""
    if mode == "ensemble":
        return _evaluate_ensemble_trial(
            cache=cache,
            model_names=model_names,
            trial_config=trial_config,
            prompt=prompt,
            artifacts_summary=artifacts_summary,
        )
    if mode == "segrank":
        if artifacts_summary is None:
            raise ValueError("SegRank tuning requires source artifacts.")
        return _evaluate_segrank_trial(
            cache=cache,
            model_names=model_names,
            trial_config=trial_config,
            selector_runtime=selector_runtime,
            artifacts_summary=artifacts_summary,
        )
    raise ValueError(f"Unsupported tuning mode: {mode}")


def run_selector_tuning(tuning_config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Run selector tuning on a validation split and return the trial leaderboard."""
    root = Path(project_root)
    mode = str(tuning_config.get("mode", "ensemble")).lower()
    prompt = str(tuning_config.get("prompt", ""))
    search_cfg = tuning_config.get("search", {})
    num_trials = int(search_cfg.get("num_trials", 20))
    include_baseline = bool(search_cfg.get("include_baseline", True))
    metric_name = str(tuning_config.get("metric", "selector_dice"))
    maximize = bool(tuning_config.get("maximize", True))
    rng = random.Random(int(search_cfg.get("random_seed", 42)))

    ensemble_config_path = root / tuning_config.get("ensemble_config", "configs/ensemble/default.yaml")
    base_ensemble_config = load_registry_config(ensemble_config_path)
    registry = build_registry(ensemble_config_path, root)
    model_names = [spec.name for spec in registry]
    device = resolve_device(str(tuning_config.get("device", "cuda")).lower())
    rows = _load_rows(root / tuning_config.get("split_csv", "datasets/agentpolyp_2504/unified_split/manifests/val.csv"), int(tuning_config.get("max_samples", 0)))
    cache, common_image_size = _build_cache(
        root_dir=root / tuning_config.get("root_dir", "datasets/agentpolyp_2504/unified_split"),
        rows=rows,
        registry=registry,
        device=device,
        batch_size=int(tuning_config.get("batch_size", 8)),
    )

    artifacts_summary = None
    anchor_source_priors = (
        base_ensemble_config.get("scoring", {})
        .get("policy", {})
        .get("anchor", {})
        .get("source_priors", {})
    )
    if mode == "segrank":
        artifacts_summary = load_source_artifacts(root / tuning_config.get("artifacts_dir", "artifacts/source_artifacts"))
    elif anchor_source_priors.get("enabled", False):
        artifacts_summary = load_source_artifacts(root / anchor_source_priors.get("artifacts_dir", "artifacts/source_artifacts"))

    trials: list[dict[str, Any]] = []
    sampled_parameter_sets: list[dict[str, Any]] = []
    if include_baseline:
        sampled_parameter_sets.append({})
    for _ in range(num_trials):
        sampled: dict[str, Any] = {}
        for name, spec in search_cfg.get("parameters", {}).items():
            sampled[name] = _sample_parameter(name=name, spec=spec, rng=rng)
        sampled_parameter_sets.append(sampled)

    for index, sampled_parameters in enumerate(sampled_parameter_sets):
        trial_config, selector_runtime = _resolve_trial_payload(
            base_config=base_ensemble_config,
            tuning_config=tuning_config,
            sampled_parameters=sampled_parameters,
        )
        summary = _evaluate_trial(
            mode=mode,
            cache=cache,
            model_names=model_names,
            trial_config=trial_config,
            selector_runtime=selector_runtime,
            prompt=prompt,
            artifacts_summary=artifacts_summary,
        )
        metric_value = float(summary["overall"][metric_name])
        trials.append(
            {
                "trial_index": index,
                "parameters": sampled_parameters,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "summary": summary,
                "effective_ensemble_config": trial_config,
                "effective_selector_runtime": selector_runtime,
            }
        )

    trials.sort(key=lambda item: item["metric_value"], reverse=maximize)
    best_trial = trials[0]
    result = {
        "mode": mode,
        "metric": metric_name,
        "maximize": maximize,
        "num_trials_evaluated": len(trials),
        "num_samples": len(cache),
        "common_image_size": common_image_size,
        "leaderboard": [
            {
                "trial_index": trial["trial_index"],
                "metric_name": trial["metric_name"],
                "metric_value": trial["metric_value"],
                "parameters": trial["parameters"],
            }
            for trial in trials
        ],
        "best_trial": best_trial,
    }

    output_dir = tuning_config.get("output_dir")
    if output_dir:
        output_path = root / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        write_json(output_path / "tuning_results.json", result)
        write_json(output_path / "best_trial.json", best_trial)
        (output_path / "best_ensemble_config.json").write_text(
            json.dumps(best_trial["effective_ensemble_config"], indent=2) + "\n",
            encoding="utf-8",
        )
        (output_path / "best_selector_runtime.json").write_text(
            json.dumps(best_trial["effective_selector_runtime"], indent=2) + "\n",
            encoding="utf-8",
        )

    return result
