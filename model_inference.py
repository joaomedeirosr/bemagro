"""Inferência com checkpoint versionado e limiar calibrado."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError

from checkpoint import load_checkpoint
from preprocessing import IMAGE_SIZE, image_transform


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            image_transform(),
        ]
    )


def process_single_image(model, file_path, output_path, device, threshold) -> None:
    try:
        with Image.open(file_path) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError):
        print(f"Arquivo inválido ignorado: {file_path}")
        return

    tensor = inference_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.softmax(model(tensor), dim=1)[:, 1]
    mask = (probability >= threshold).squeeze().cpu().numpy()
    output_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)
    print(f"Imagem segmentada salva em: {output_path}")


def segment_images(model, image_path, output, device, threshold) -> None:
    image_path = Path(image_path)
    output = Path(output)
    if image_path.is_dir():
        output.mkdir(parents=True, exist_ok=True)
        for file_path in sorted(image_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                process_single_image(
                    model,
                    file_path,
                    output / f"seg_{file_path.name}",
                    device,
                    threshold,
                )
        return

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato de imagem não suportado: {image_path}")
    process_single_image(model, image_path, output, device, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência de vegetação com U-Net v2.")
    parser.add_argument("--rgb", required=True, help="Imagem RGB ou diretório de blocos.")
    parser.add_argument("--modelpath", required=True, help="Checkpoint v2.")
    parser.add_argument("--output", required=True, help="Arquivo ou diretório de saída.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Sobrescreve o limiar calibrado salvo no checkpoint.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold deve estar entre 0 e 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA foi solicitada, mas não está disponível")

    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    model, checkpoint = load_checkpoint(args.modelpath, device)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint["inference_threshold"])
    )
    print(f"Dispositivo: {device}; limiar: {threshold:.2f}")
    segment_images(model, args.rgb, args.output, device, threshold)


if __name__ == "__main__":
    main()
