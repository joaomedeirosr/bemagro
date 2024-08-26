"""
Propósito: Criacao do Dataloader realiza o carregamento do dataset para treino.
Autor : Joao Victor Rocha <jvmedeirosr@gmail.com>
"""
import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageSegmentationDataset(Dataset):
    """
    Dataset para carregar imagens e máscaras de segmentação para treinamento.
    """
    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple:
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        mask = torch.squeeze(mask, dim=0)


        return image, mask
