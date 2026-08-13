"""
Criação do Dataset responsável por carregar pares de imagem e máscara.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class ImageSegmentationDataset(Dataset):
    """Dataset de segmentação binária com pareamento explícito por nome-base.

    ``joint_transform`` deve receber ``image`` e ``mask`` como arrays NumPy e
    retornar um dicionário com as mesmas chaves. Esse é o contrato usado pelo
    Albumentations e garante que transformações geométricas aleatórias usem os
    mesmos parâmetros na imagem e na máscara. Transformações que alteram o tipo
    da imagem, como normalização, devem ser passadas em ``transform``.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.joint_transform = joint_transform
        self.samples = self._build_pairs()

    @staticmethod
    def _index_files(directory: Path) -> Dict[str, Path]:
        if not directory.is_dir():
            raise NotADirectoryError(f"Diretório não encontrado: {directory}")

        indexed: Dict[str, Path] = {}
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            key = path.stem
            if key in indexed:
                raise ValueError(
                    f"Mais de um arquivo com o nome-base '{key}' em {directory}"
                )
            indexed[key] = path

        return indexed

    def _build_pairs(self) -> List[Tuple[Path, Path]]:
        images = self._index_files(self.image_dir)
        masks = self._index_files(self.mask_dir)

        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        if missing_masks or missing_images:
            details = []
            if missing_masks:
                details.append(f"sem máscara: {', '.join(missing_masks[:5])}")
            if missing_images:
                details.append(f"sem imagem: {', '.join(missing_images[:5])}")
            raise ValueError(
                "Pareamento inválido entre imagens e máscaras ("
                + "; ".join(details)
                + ")"
            )

        if not images:
            raise ValueError(
                f"Nenhum par de imagem e máscara foi encontrado em "
                f"{self.image_dir} e {self.mask_dir}"
            )

        return [(images[key], masks[key]) for key in sorted(images)]

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _mask_to_classes(mask: np.ndarray, mask_path: Path) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError(f"A máscara deve ter um único canal: {mask_path}")

        unique_values = np.unique(mask)
        if np.isin(unique_values, (0, 1)).all():
            class_mask = mask.astype(np.int64)
        elif np.isin(unique_values, (0, 255)).all():
            class_mask = (mask == 255).astype(np.int64)
        else:
            values = ", ".join(map(str, unique_values[:10]))
            raise ValueError(
                f"Máscara binária inválida em {mask_path}. "
                f"Esperado 0/1 ou 0/255; encontrado: {values}"
            )

        return torch.from_numpy(class_mask.copy())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        with Image.open(mask_path) as source_mask:
            mask = source_mask.convert("L")

        if image.size != mask.size:
            raise ValueError(
                f"Dimensões diferentes no par '{image_path.stem}': "
                f"imagem={image.size}, máscara={mask.size}"
            )

        image_array = np.asarray(image)
        mask_array = np.asarray(mask)

        if self.joint_transform:
            transformed = self.joint_transform(image=image_array, mask=mask_array)
            image_array = transformed["image"]
            mask_array = transformed["mask"]

        image = Image.fromarray(image_array)
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1).float() / 255.0

        mask_tensor = self._mask_to_classes(mask_array, mask_path)
        return image_tensor, mask_tensor
