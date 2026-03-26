"""Shared neural network building blocks for segmentation architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(num_channels: int, norm: str = "bn") -> nn.Module:
    """Create a normalization layer matching the requested style."""
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "gn":
        groups = 8 if num_channels >= 8 else 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm: {norm}")


class ConvBlock(nn.Module):
    """Apply two convolution-normalization-activation stages."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn") -> None:
        """Initialize the double-convolution block."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the input feature map with the configured block."""
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """Residual convolutional block with optional downsampling and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: str = "gn",
        dropout: float = 0.0,
    ) -> None:
        """Initialize the residual path and shortcut connection."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = make_norm(out_channels, norm)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_norm(out_channels, norm)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_norm(out_channels, norm),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block and return the fused activation."""
        residual = self.shortcut(x)
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution followed by normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        norm: str = "bn",
    ) -> None:
        """Initialize the separable convolution block."""
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = make_norm(out_channels, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the depthwise and pointwise convolutions."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class UpBlock(nn.Module):
    """Upsample decoder features, concatenate skip features, and refine them."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm: str = "bn") -> None:
        """Initialize the decoder upsampling block."""
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample decoder features and fuse them with the skip tensor."""
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling for multi-scale context aggregation."""

    def __init__(self, in_channels: int, out_channels: int, rates: tuple[int, ...] = (1, 6, 12, 18)) -> None:
        """Initialize ASPP branches and the projection head."""
        super().__init__()
        branches = []
        for rate in rates:
            if rate == 1:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                        make_norm(out_channels, "bn"),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            padding=rate,
                            dilation=rate,
                            bias=False,
                        ),
                        make_norm(out_channels, "bn"),
                        nn.ReLU(inplace=True),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            make_norm(out_channels, "bn"),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, kernel_size=1, bias=False),
            make_norm(out_channels, "bn"),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate multi-scale context features for the input tensor."""
        pooled = self.pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        outputs = [branch(x) for branch in self.branches]
        outputs.append(pooled)
        return self.project(torch.cat(outputs, dim=1))
