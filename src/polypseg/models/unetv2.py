"""Paper-style U-Net v2 with a PVTv2 encoder and SDI skip fusion."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

from .checkpointing import extract_state_dict
from .pvtv2 import pvt_v2_b2


class ResNetEncoder(nn.Module):
    """Torchvision ResNet encoder retained for local DeepLabV3+ usage."""

    def __init__(self, in_channels: int = 3, backbone: str = "resnet34", pretrained: bool = False) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("ImageNet-pretrained ResNet encoder currently requires in_channels=3.")

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            encoder = resnet18(weights=weights)
            widths = (64, 64, 128, 256, 512)
        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            encoder = resnet34(weights=weights)
            widths = (64, 64, 128, 256, 512)
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone}")

        self.widths = widths
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


def _load_matching_state_dict(module: nn.Module, checkpoint_path: str | None, label: str) -> None:
    """Load matching parameters from an external checkpoint without requiring exact key parity."""
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    if not path.is_file():
        warnings.warn(f"{label} checkpoint path does not exist: {checkpoint_path}")
        return

    payload = torch.load(path, map_location="cpu")
    state_dict = extract_state_dict(payload)
    if label == "backbone":
        stripped = {
            key[len("backbone.") :]: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        if stripped:
            state_dict = stripped
    model_state = module.state_dict()
    matched = {key: value for key, value in state_dict.items() if key in model_state and model_state[key].shape == value.shape}
    if not matched:
        warnings.warn(f"No compatible {label} weights were found in: {checkpoint_path}")
        return

    model_state.update(matched)
    module.load_state_dict(model_state)
    print(f"[UNetV2] loaded {len(matched)} {label} tensors from {checkpoint_path}", flush=True)


class ChannelAttention(nn.Module):
    """CBAM-style channel attention used in the original implementation."""

    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention used in the original implementation."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class BasicConv2d(nn.Module):
    """Conv-BN block matching the upstream implementation."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class Encoder(nn.Module):
    """PVTv2-B2 backbone wrapper that exposes four multi-scale feature maps."""

    def __init__(self, pretrained_path: str | None = None) -> None:
        super().__init__()
        self.backbone = pvt_v2_b2()
        if pretrained_path:
            _load_matching_state_dict(self.backbone, pretrained_path, label="backbone")
        else:
            warnings.warn("No PVTv2 pretrained path provided for UNetV2; encoder will use random initialization.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1, f2, f3, f4 = self.backbone(x)
        return f1, f2, f3, f4


class SDI(nn.Module):
    """Scale-aware skip fusion block from the U-Net v2 repo."""

    def __init__(self, channel: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs: list[torch.Tensor], anchor: torch.Tensor) -> torch.Tensor:
        fused = torch.ones_like(anchor)
        target_h, target_w = anchor.shape[-2:]
        for index, x in enumerate(xs):
            if x.shape[-2:] != (target_h, target_w):
                if x.shape[-2] > target_h or x.shape[-1] > target_w:
                    x = F.adaptive_avg_pool2d(x, (target_h, target_w))
                else:
                    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=True)
            fused = fused * self.convs[index](x)
        return fused


class UNetV2(nn.Module):
    """Faithful local adaptation of the paper/repo U-Net v2 architecture."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channel: int = 32,
        deep_supervision: bool = True,
        pretrained_path: str | None = None,
        checkpoint_path: str | None = None,
        n_classes: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("UNetV2 with PVTv2-B2 expects in_channels=3.")

        self.num_classes = n_classes if n_classes is not None else num_classes
        self.deep_supervision = deep_supervision
        self.encoder = Encoder(pretrained_path=pretrained_path)

        self.ca_1 = ChannelAttention(64)
        self.sa_1 = SpatialAttention()
        self.ca_2 = ChannelAttention(128)
        self.sa_2 = SpatialAttention()
        self.ca_3 = ChannelAttention(320)
        self.sa_3 = SpatialAttention()
        self.ca_4 = ChannelAttention(512)
        self.sa_4 = SpatialAttention()

        self.Translayer_1 = BasicConv2d(64, channel, kernel_size=1)
        self.Translayer_2 = BasicConv2d(128, channel, kernel_size=1)
        self.Translayer_3 = BasicConv2d(320, channel, kernel_size=1)
        self.Translayer_4 = BasicConv2d(512, channel, kernel_size=1)

        self.sdi_1 = SDI(channel)
        self.sdi_2 = SDI(channel)
        self.sdi_3 = SDI(channel)
        self.sdi_4 = SDI(channel)

        self.seg_outs = nn.ModuleList([nn.Conv2d(channel, self.num_classes, kernel_size=1, stride=1) for _ in range(4)])

        self.deconv2 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)

        if checkpoint_path:
            _load_matching_state_dict(self, checkpoint_path, label="model")

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        seg_outs: list[torch.Tensor] = []
        f1, f2, f3, f4 = self.encoder(x)

        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)

        f41 = self.sdi_4([f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f1, f2, f3, f4], f2)
        f11 = self.sdi_1([f1, f2, f3, f4], f1)

        seg_outs.append(self.seg_outs[0](f41))
        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y))
        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))
        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))

        for index, output in enumerate(seg_outs):
            seg_outs[index] = F.interpolate(output, scale_factor=4, mode="bilinear", align_corners=False)

        if self.training and self.deep_supervision:
            return seg_outs[::-1]
        return seg_outs[-1]
