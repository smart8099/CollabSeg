"""Public exports for SegRank offline artifact utilities."""

from .artifacts import ensure_dir, write_json
from .descriptors import aggregate_image_descriptors, compute_image_descriptor
from .embeddings import aggregate_embedding_vectors, cosine_similarity, embedding_distance
from .evidence import aggregate_evidence, compute_prediction_evidence
from .morphology import aggregate_mask_morphology, compute_mask_morphology
from .types import DatasetDescriptor, EvidenceSummary, ModelDatasetArtifact, MorphologyDescriptor

__all__ = [
    "DatasetDescriptor",
    "EvidenceSummary",
    "ModelDatasetArtifact",
    "MorphologyDescriptor",
    "aggregate_evidence",
    "aggregate_embedding_vectors",
    "aggregate_image_descriptors",
    "aggregate_mask_morphology",
    "cosine_similarity",
    "compute_image_descriptor",
    "compute_mask_morphology",
    "compute_prediction_evidence",
    "embedding_distance",
    "ensure_dir",
    "write_json",
]
