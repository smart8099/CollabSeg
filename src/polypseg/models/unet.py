"""Implementation of a baseline U-Net segmentation model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock, UpBlock


class UNet(nn.Module):
    """Standard encoder-decoder U-Net with skip connections."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
        norm: str = "bn",
    ) -> None:
        """Initialize the U-Net encoder, bottleneck, decoder, and head."""
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for feature in features:
            self.encoders.append(ConvBlock(prev_channels, feature, norm=norm))
            prev_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, norm=norm)

        reversed_features = list(reversed(features))
        self.decoders = nn.ModuleList()
        decoder_in = features[-1] * 2
        for feature in reversed_features:
            self.decoders.append(UpBlock(decoder_in, feature, feature, norm=norm))
            decoder_in = feature

        self.head = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return segmentation logits."""
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return self.head(x)
