"""Implementation of a nested skip-connection U-Net++ model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock


class UNetPlusPlus(nn.Module):
    """U-Net++ with dense nested skip connections and optional deep supervision."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
        deep_supervision: bool = False,
        norm: str = "bn",
    ) -> None:
        """Initialize the nested encoder-decoder graph for U-Net++."""
        super().__init__()
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)
        f0, f1, f2, f3 = features

        self.x00 = ConvBlock(in_channels, f0, norm=norm)
        self.x10 = ConvBlock(f0, f1, norm=norm)
        self.x20 = ConvBlock(f1, f2, norm=norm)
        self.x30 = ConvBlock(f2, f3, norm=norm)
        self.x40 = ConvBlock(f3, f3 * 2, norm=norm)

        self.x01 = ConvBlock(f0 + f1, f0, norm=norm)
        self.x11 = ConvBlock(f1 + f2, f1, norm=norm)
        self.x21 = ConvBlock(f2 + f3, f2, norm=norm)
        self.x31 = ConvBlock(f3 + f3 * 2, f3, norm=norm)

        self.x02 = ConvBlock(f0 * 2 + f1, f0, norm=norm)
        self.x12 = ConvBlock(f1 * 2 + f2, f1, norm=norm)
        self.x22 = ConvBlock(f2 * 2 + f3, f2, norm=norm)

        self.x03 = ConvBlock(f0 * 3 + f1, f0, norm=norm)
        self.x13 = ConvBlock(f1 * 3 + f2, f1, norm=norm)

        self.x04 = ConvBlock(f0 * 4 + f1, f0, norm=norm)

        self.final = nn.Conv2d(f0, num_classes, kernel_size=1)
        if deep_supervision:
            self.ds_heads = nn.ModuleList([nn.Conv2d(f0, num_classes, kernel_size=1) for _ in range(4)])

    def _up_cat(self, left: list[torch.Tensor], right: torch.Tensor) -> torch.Tensor:
        """Upsample one tensor and concatenate it with skip tensors."""
        target_size = left[0].shape[-2:]
        upsampled = F.interpolate(right, size=target_size, mode="bilinear", align_corners=False)
        return torch.cat([*left, upsampled], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run the nested decoder and return segmentation logits."""
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        x01 = self.x01(self._up_cat([x00], x10))
        x11 = self.x11(self._up_cat([x10], x20))
        x21 = self.x21(self._up_cat([x20], x30))
        x31 = self.x31(self._up_cat([x30], x40))

        x02 = self.x02(self._up_cat([x00, x01], x11))
        x12 = self.x12(self._up_cat([x10, x11], x21))
        x22 = self.x22(self._up_cat([x20, x21], x31))

        x03 = self.x03(self._up_cat([x00, x01, x02], x12))
        x13 = self.x13(self._up_cat([x10, x11, x12], x22))

        x04 = self.x04(self._up_cat([x00, x01, x02, x03], x13))

        if self.deep_supervision:
            return [head(node) for head, node in zip(self.ds_heads, [x01, x02, x03, x04])]
        return self.final(x04)
