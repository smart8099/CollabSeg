"""Implementation of a lightweight DeepLabV3+ style segmentation model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ASPP, ResidualConvBlock, SeparableConv2d
from .unetv2 import ResNetEncoder

try:
    import timm
except ModuleNotFoundError:
    timm = None


class SimpleEncoder(nn.Module):
    """Simple residual encoder used by the local DeepLabV3+ implementation."""

    def __init__(self, in_channels: int = 3, widths: tuple[int, ...] = (64, 128, 256, 512)) -> None:
        """Initialize the encoder stem and residual stages."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualConvBlock(widths[0], widths[0], norm="bn")
        self.layer2 = ResidualConvBlock(widths[0], widths[1], stride=2, norm="bn")
        self.layer3 = ResidualConvBlock(widths[1], widths[2], stride=2, norm="bn")
        self.layer4 = ResidualConvBlock(widths[2], widths[3], stride=2, norm="bn")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return low-level and high-level encoder features."""
        x = self.stem(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)
        return low_level, x


class XceptionEncoder(nn.Module):
    """timm-backed Xception feature extractor for DeepLabV3+."""

    def __init__(self, backbone: str = "xception65", pretrained: bool = False) -> None:
        super().__init__()
        if timm is None:
            raise ModuleNotFoundError("Xception encoder requires timm to be installed.")
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 4),
        )
        self.widths = tuple(self.backbone.feature_info.channels())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low_level, high_level = self.backbone(x)
        return low_level, high_level


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ style segmentation model with ASPP and decoder refinement."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_widths: tuple[int, ...] = (64, 128, 256, 512),
        aspp_channels: int = 256,
        decoder_channels: int = 128,
        encoder_name: str = "custom",
        encoder_pretrained: bool = False,
    ) -> None:
        """Initialize the encoder, ASPP module, decoder, and prediction head."""
        super().__init__()
        self.encoder_name = encoder_name.lower()

        if self.encoder_name == "custom":
            self.encoder = SimpleEncoder(in_channels=in_channels, widths=encoder_widths)
            low_level_channels = encoder_widths[0]
            high_level_channels = encoder_widths[-1]
        elif self.encoder_name in {"resnet18", "resnet34"}:
            self.encoder = ResNetEncoder(
                in_channels=in_channels,
                backbone=self.encoder_name,
                pretrained=encoder_pretrained,
            )
            # layer1 output → low-level features, layer4 output → high-level features
            low_level_channels = self.encoder.widths[1]
            high_level_channels = self.encoder.widths[4]
        elif self.encoder_name in {"xception", "xception65", "legacy_xception"}:
            if in_channels != 3:
                raise ValueError("Xception encoder currently requires in_channels=3.")
            backbone_name = "xception65" if self.encoder_name == "xception" else self.encoder_name
            self.encoder = XceptionEncoder(backbone=backbone_name, pretrained=encoder_pretrained)
            low_level_channels, high_level_channels = self.encoder.widths
        else:
            raise ValueError(f"Unsupported encoder_name: {encoder_name}")

        self.aspp = ASPP(high_level_channels, aspp_channels)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            SeparableConv2d(aspp_channels + 48, decoder_channels, norm="bn"),
            SeparableConv2d(decoder_channels, decoder_channels, norm="bn"),
        )
        self.head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and upsample logits to the input size."""
        spatial_size = x.shape[-2:]

        if self.encoder_name == "custom":
            low_level, encoder_out = self.encoder(x)
        elif self.encoder_name in {"xception", "xception65", "legacy_xception"}:
            low_level, encoder_out = self.encoder(x)
        else:
            _, low_level, _, _, encoder_out = self.encoder(x)

        x = self.aspp(encoder_out)
        x = F.interpolate(x, size=low_level.shape[-2:], mode="bilinear", align_corners=False)
        low_level = self.low_level_proj(low_level)
        x = self.decoder(torch.cat([x, low_level], dim=1))
        x = self.head(x)
        return F.interpolate(x, size=spatial_size, mode="bilinear", align_corners=False)
