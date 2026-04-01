"""Dataclasses used by the SegRank offline artifact pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatasetDescriptor:
    """Summarize image-distribution characteristics for one dataset."""

    num_samples: int
    rgb_mean: list[float]
    rgb_std: list[float]
    grayscale_mean: float
    grayscale_std: float
    grayscale_entropy: float
    edge_density: float
    embedding: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


@dataclass
class MorphologyDescriptor:
    """Summarize geometric properties of binary masks for one dataset."""

    num_samples: int
    mean_area_ratio: float
    std_area_ratio: float
    mean_component_count: float
    mean_boundary_ratio: float
    mean_bbox_aspect_ratio: float
    mean_centroid_x: float
    mean_centroid_y: float
    mean_empty_mask_ratio: float
    embedding: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


@dataclass
class EvidenceSummary:
    """Aggregate model evidence features over one source dataset."""

    num_samples: int
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_stds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


@dataclass
class ModelDatasetArtifact:
    """Store model behavior and quality for one source dataset."""

    model_name: str
    source_dataset: str
    num_samples: int
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    evidence: EvidenceSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        payload = asdict(self)
        payload["evidence"] = self.evidence.to_dict()
        return payload
