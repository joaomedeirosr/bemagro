"""Formato versionado de checkpoint do pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model import ARCHITECTURE_NAME, UNetResNet
from preprocessing import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


CHECKPOINT_VERSION = 2


def build_checkpoint(
    model: UNetResNet,
    epoch: int,
    metrics: dict[str, float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "n_classes": 2,
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "inference_threshold": float(threshold),
        "epoch": int(epoch),
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[UNetResNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            "Checkpoint legado ou incompatível. Treine novamente com a arquitetura v2."
        )
    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        raise ValueError(f"Arquitetura incompatível: {checkpoint.get('architecture')}")
    if checkpoint.get("image_size") != IMAGE_SIZE or checkpoint.get("normalization") != {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }:
        raise ValueError("Pré-processamento do checkpoint incompatível com esta versão")

    model = UNetResNet(n_classes=int(checkpoint["n_classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint
