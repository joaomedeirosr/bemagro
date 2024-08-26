import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class UNetResNet(nn.Module):
    """
    U-Net com backbone ResNet-50 para segmentação de imagens.
    """
    def __init__(self, n_classes: int):
        super(UNetResNet, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder_layers = list(self.resnet.children())[:-2]  # Remover a camada fully connected
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.conv_final(x)
        return x