"""Treinamento reproduzível da U-Net/ResNet-50."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from pathlib import Path

import albumentations as A
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from checkpoint import build_checkpoint
from dataset import ImageSegmentationDataset
from model import UNetResNet
from preprocessing import IMAGE_SIZE, image_transform
from utils import plot_training_metrics


BLOCK_PATTERN = re.compile(r"^bloco_(\d+)_(\d+)$")
METRIC_NAMES = (
    "loss",
    "background_iou",
    "vegetation_iou",
    "mean_iou",
    "dice",
    "precision",
    "recall",
    "accuracy",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spatial_split(samples, validation_fraction: float) -> tuple[list[int], list[int]]:
    """Reserva colunas inteiras da direita para evitar vazamento espacial."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction deve estar entre 0 e 1")

    coordinates = []
    for index, (image_path, _) in enumerate(samples):
        match = BLOCK_PATTERN.fullmatch(image_path.stem)
        if not match:
            raise ValueError(
                "O split espacial exige nomes no formato bloco_X_Y: "
                f"{image_path.name}"
            )
        coordinates.append((index, int(match.group(1))))

    columns = sorted({column for _, column in coordinates}, reverse=True)
    if len(columns) < 2:
        raise ValueError("São necessárias pelo menos duas colunas para o split espacial")

    target = max(1, math.ceil(len(samples) * validation_fraction))
    validation_columns: set[int] = set()
    selected = 0
    for column in columns[:-1]:
        validation_columns.add(column)
        selected += sum(candidate == column for _, candidate in coordinates)
        if selected >= target:
            break

    validation = [i for i, column in coordinates if column in validation_columns]
    training = [i for i, column in coordinates if column not in validation_columns]
    if not training or not validation:
        raise ValueError("O split espacial produziu um conjunto vazio")
    return training, validation


def train_joint_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.12, 0.12, p=0.35),
            A.HueSaturationValue(8, 12, 8, p=0.25),
        ]
    )


def evaluation_joint_transform() -> A.Compose:
    return A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv.INTER_LINEAR)])


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.cross_entropy(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
        dimensions = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dimensions)
        denominator = probabilities.sum(dimensions) + one_hot.sum(dimensions)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return 0.5 * ce + 0.5 * dice_loss


def class_weights(dataset: ImageSegmentationDataset, indexes: list[int], device) -> torch.Tensor:
    counts = np.zeros(2, dtype=np.int64)
    for index in indexes:
        mask_path = dataset.samples[index][1]
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert("L")) >= 128
        counts[1] += int(mask.sum())
        counts[0] += int(mask.size - mask.sum())
    frequencies = counts / counts.sum()
    weights = 1.0 / np.sqrt(np.maximum(frequencies, 1e-8))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def update_confusion(confusion: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor) -> None:
    indexes = (targets * 2 + predictions).reshape(-1)
    confusion += torch.bincount(indexes.detach().cpu(), minlength=4).reshape(2, 2)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, float]:
    tn, fp, fn, tp = confusion.flatten().double().tolist()
    background_iou = tn / (tn + fp + fn) if tn + fp + fn else 1.0
    vegetation_iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    total = tn + fp + fn + tp
    return {
        "background_iou": background_iou,
        "vegetation_iou": vegetation_iou,
        "mean_iou": (background_iou + vegetation_iou) / 2,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": (tn + tp) / total if total else 0.0,
    }


def evaluate(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, masks).item() * images.size(0)
            total_samples += images.size(0)
            update_confusion(confusion, logits.argmax(dim=1), masks)
    return {"loss": total_loss / total_samples, **metrics_from_confusion(confusion)}


def calibrate_threshold(model, loader, device) -> tuple[float, float]:
    probabilities = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            probabilities.append(torch.softmax(model(images), dim=1)[:, 1].cpu())
            targets.append(masks.bool().cpu())
    probabilities = torch.cat(probabilities).reshape(-1)
    targets = torch.cat(targets).reshape(-1)

    best_threshold = 0.5
    best_mean_iou = -1.0
    for threshold in np.arange(0.10, 0.901, 0.05):
        prediction = probabilities >= float(threshold)
        confusion = torch.zeros((2, 2), dtype=torch.int64)
        update_confusion(confusion, prediction.long(), targets.long())
        mean_iou = metrics_from_confusion(confusion)["mean_iou"]
        if mean_iou > best_mean_iou:
            best_threshold = float(round(threshold, 2))
            best_mean_iou = mean_iou
    return best_threshold, best_mean_iou


def make_loader(dataset, indexes, batch_size, shuffle, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indexes),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def main(args) -> None:
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas não está disponível")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    print(f"Dispositivo: {device}")

    evaluation_dataset = ImageSegmentationDataset(
        args.rgb,
        args.groundtruth,
        transform=image_transform(),
        joint_transform=evaluation_joint_transform(),
    )
    training_indexes, validation_indexes = spatial_split(
        evaluation_dataset.samples, args.validation_fraction
    )
    training_dataset = ImageSegmentationDataset(
        args.rgb,
        args.groundtruth,
        transform=image_transform(),
        joint_transform=train_joint_transform(),
    )
    print(
        f"Split espacial: {len(training_indexes)} treino, "
        f"{len(validation_indexes)} validação"
    )

    train_loader = make_loader(training_dataset, training_indexes, args.batch_size, True, args.num_workers, args.seed)
    train_eval_loader = make_loader(evaluation_dataset, training_indexes, args.batch_size, False, args.num_workers, args.seed)
    validation_loader = make_loader(evaluation_dataset, validation_indexes, args.batch_size, False, args.num_workers, args.seed)

    model = UNetResNet(n_classes=2, pretrained=True).to(device)
    weights = class_weights(evaluation_dataset, training_indexes, device)
    print(f"Pesos das classes [fundo, vegetação]: {weights.tolist()}")
    criterion = CombinedSegmentationLoss(weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    model_path = Path(args.modelpath)
    metrics_path = Path(args.metrics_output or model_path.with_name(model_path.stem + "_metrics.csv"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "learning_rate"] + [f"train_{name}" for name in METRIC_NAMES] + [f"validation_{name}" for name in METRIC_NAMES]
    rows: list[dict[str, float | int]] = []
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    best_mean_iou = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (images, masks) in enumerate(tqdm(train_loader, desc=f"Época {epoch}/{args.epochs}"), 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = criterion(model(images), masks) / args.accumulation_steps
            scaler.scale(loss).backward()
            if step % args.accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        train_metrics = evaluate(model, train_eval_loader, criterion, device)
        validation_metrics = evaluate(model, validation_loader, criterion, device)
        scheduler.step(validation_metrics["loss"])
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        rows.append(row)
        with metrics_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=fieldnames).writerow(row)

        print(
            f"Época {epoch}: train_mIoU={train_metrics['mean_iou']:.4f}, "
            f"val_mIoU={validation_metrics['mean_iou']:.4f}, "
            f"val_loss={validation_metrics['loss']:.4f}"
        )
        if validation_metrics["mean_iou"] > best_mean_iou + args.min_delta:
            best_mean_iou = validation_metrics["mean_iou"]
            epochs_without_improvement = 0
            torch.save(build_checkpoint(model, epoch, validation_metrics), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping na época {epoch}")
                break

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    threshold, calibrated_miou = calibrate_threshold(model, validation_loader, device)
    checkpoint["inference_threshold"] = threshold
    checkpoint["calibrated_validation_mean_iou"] = calibrated_miou
    torch.save(checkpoint, model_path)
    print(f"Melhor limiar de validação: {threshold:.2f} (mIoU={calibrated_miou:.4f})")

    plot_training_metrics(rows, metrics_path.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina a U-Net/ResNet-50 v2.")
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--modelpath", required=True)
    parser.add_argument("--metrics-output")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    main(parser.parse_args())
