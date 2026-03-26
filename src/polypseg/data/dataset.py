"""Dataset and transform utilities for polyp segmentation experiments."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _to_tensors(
    image: Image.Image,
    mask: Image.Image,
    mean: list[float],
    std: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a PIL image and mask into normalized tensors."""
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = (image_np - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))

    mask_np = np.asarray(mask, dtype=np.float32)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    mask_tensor = torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)
    return image_tensor, mask_tensor


def build_eval_transforms(
    image_size: int,
    mean: list[float],
    std: list[float],
) -> Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]]:
    """Build deterministic evaluation transforms for image-mask pairs."""
    def transform(image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize inputs for evaluation and convert them to tensors."""
        image = image.resize((image_size, image_size), Image.BILINEAR)
        mask = mask.resize((image_size, image_size), Image.NEAREST)
        return _to_tensors(image, mask, mean, std)

    return transform


def build_train_transforms(
    image_size: int,
    mean: list[float],
    std: list[float],
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.0,
    rotate90_prob: float = 0.5,
    color_jitter_prob: float = 0.3,
    brightness: float = 0.15,
    contrast: float = 0.15,
) -> Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]]:
    """Build paired training transforms with light augmentation."""
    def transform(image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply paired augmentations to the image and mask before tensor conversion."""
        image = image.resize((image_size, image_size), Image.BILINEAR)
        mask = mask.resize((image_size, image_size), Image.NEAREST)

        if random.random() < horizontal_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < vertical_flip_prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < rotate90_prob:
            rotations = random.choice([1, 2, 3])
            image = image.transpose(
                {
                    1: Image.ROTATE_90,
                    2: Image.ROTATE_180,
                    3: Image.ROTATE_270,
                }[rotations]
            )
            mask = mask.transpose(
                {
                    1: Image.ROTATE_90,
                    2: Image.ROTATE_180,
                    3: Image.ROTATE_270,
                }[rotations]
            )
        if random.random() < color_jitter_prob:
            image_np = np.asarray(image, dtype=np.float32)
            brightness_scale = 1.0 + random.uniform(-brightness, brightness)
            contrast_scale = 1.0 + random.uniform(-contrast, contrast)
            image_np = np.clip(image_np * brightness_scale, 0, 255)
            mean_pixel = image_np.mean(axis=(0, 1), keepdims=True)
            image_np = np.clip((image_np - mean_pixel) * contrast_scale + mean_pixel, 0, 255)
            image = Image.fromarray(image_np.astype(np.uint8))

        return _to_tensors(image, mask, mean, std)

    return transform


class PolypSegmentationDataset(Dataset):
    """CSV-backed dataset for image-mask polyp segmentation samples."""

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        transform: Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Load dataset rows and store the transform callable."""
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.transform = transform
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.rows = list(reader)

    def __len__(self) -> int:
        """Return the number of samples listed in the manifest."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """Load and return one image-mask sample with metadata."""
        row = self.rows[index]
        image_path = self.root_dir / row["image_path"]
        mask_path = self.root_dir / row["mask_path"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image_tensor, mask_tensor = self.transform(image, mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": row["sample_id"],
            "source_dataset": row["source_dataset"],
        }
