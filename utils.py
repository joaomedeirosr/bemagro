"""Visualização das métricas do treinamento."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_metrics(rows: list[dict], output_path: str | Path) -> None:
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="treino")
    axes[0].plot(epochs, [row["validation_loss"] for row in rows], label="validação")
    axes[0].set(title="Perda", xlabel="Época", ylabel="Loss")
    axes[1].plot(epochs, [row["train_mean_iou"] for row in rows], label="treino")
    axes[1].plot(epochs, [row["validation_mean_iou"] for row in rows], label="validação")
    axes[1].set(title="mIoU", xlabel="Época", ylabel="mIoU", ylim=(0, 1))
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
