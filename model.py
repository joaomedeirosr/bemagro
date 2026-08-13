"""U-Net com encoder ResNet-50 para segmentação semântica."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


ARCHITECTURE_NAME = "unet-resnet50-v2"


class ConvBlock(nn.Sequential):
    """Duas convoluções que refinam os atributos após cada skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Aumenta a resolução, concatena o skip do encoder e refina o resultado."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.refine = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat((x, skip), dim=1))


class UNetResNet(nn.Module):
    """U-Net real com cinco escalas e encoder ResNet-50 pré-treinado."""

    def __init__(self, n_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.center = ConvBlock(2048, 512)
        self.decoder4 = DecoderBlock(512, 1024, 512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        self.full_resolution = ConvBlock(64, 32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        skip1 = self.stem(x)                 # 1/2
        skip2 = self.layer1(self.pool(skip1)) # 1/4
        skip3 = self.layer2(skip2)            # 1/8
        skip4 = self.layer3(skip3)            # 1/16
        encoded = self.layer4(skip4)           # 1/32

        decoded = self.center(encoded)
        decoded = self.decoder4(decoded, skip4)
        decoded = self.decoder3(decoded, skip3)
        decoded = self.decoder2(decoded, skip2)
        decoded = self.decoder1(decoded, skip1)
        decoded = F.interpolate(decoded, size=input_size, mode="bilinear", align_corners=False)
        return self.classifier(self.full_resolution(decoded))
