"""Implementation of a modernized residual U-Net variant."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualConvBlock, SeparableConv2d


class SEBlock(nn.Module):
    """Squeeze-and-excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize the squeeze-and-excitation projection layers."""
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reweight the input tensor with learned channel attention."""
        return x * self.fc(self.pool(x))


class DecoderV2Block(nn.Module):
    """Decoder block with separable convolutions and channel attention."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Initialize the decoder fusion block."""
        super().__init__()
        self.fuse = nn.Sequential(
            SeparableConv2d(in_channels + skip_channels, out_channels, norm="gn"),
            SeparableConv2d(out_channels, out_channels, norm="gn"),
        )
        self.attn = SEBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample decoder features, fuse them with skips, and refine them."""
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.attn(x)


class UNetV2(nn.Module):
    """Residual U-Net style model with group normalization and attention."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.1,
    ) -> None:
        """Initialize the encoder, bottleneck, decoder, and prediction head."""
        super().__init__()
        self.stem = ResidualConvBlock(in_channels, features[0], norm="gn", dropout=dropout)
        self.enc2 = ResidualConvBlock(features[0], features[1], stride=2, norm="gn", dropout=dropout)
        self.enc3 = ResidualConvBlock(features[1], features[2], stride=2, norm="gn", dropout=dropout)
        self.enc4 = ResidualConvBlock(features[2], features[3], stride=2, norm="gn", dropout=dropout)
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(features[3], features[3] * 2, stride=2, norm="gn", dropout=dropout),
            SEBlock(features[3] * 2),
        )

        self.dec4 = DecoderV2Block(features[3] * 2, features[3], features[3])
        self.dec3 = DecoderV2Block(features[3], features[2], features[2])
        self.dec2 = DecoderV2Block(features[2], features[1], features[1])
        self.dec1 = DecoderV2Block(features[1], features[0], features[0])

        self.head = nn.Sequential(
            SeparableConv2d(features[0], features[0], norm="gn"),
            nn.Conv2d(features[0], num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return segmentation logits."""
        x1 = self.stem(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(x4)

        x = self.dec4(x5, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        return self.head(x)
