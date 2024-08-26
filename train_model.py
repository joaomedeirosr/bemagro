"""
Propósito: Realizar o treinamento de um modelo utilzando U-net Resnet-50 como backbone
Autor : Joao Victor Rocha <jvmedeirosr@gmail.com>
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNetResNet
from dataset import ImageSegmentationDataset
import torchvision.transforms as transforms
import numpy as np
from utils import compute_iou, compute_accuracy, plot_training_loss


def main(args) -> None:
    # Configurações de dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transformações de imagem
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Carregar o dataset
    dataset = ImageSegmentationDataset(args.rgb, args.groundtruth, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Define o modelo, função de perda e otimizador
    model = UNetResNet(n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    training_loss = []

    # Treinamento do modelo
    model.train()
    for epoch in range(50):  # Número de épocas 
        epoch_loss = 0
        iou_score = 0
        accuracy_score = 0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iou_score += compute_iou(outputs, masks)
            accuracy_score += compute_accuracy(outputs,masks)
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            avg_iou_score = iou_score / len(dataloader)
            avg_accuracy_score = accuracy_score / len(dataloader)
            training_loss.append(avg_epoch_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss}, IoU: {avg_iou_score}, Accuracy: {avg_accuracy_score} ")
    
    # Plotar a Loss do treino após o término do treinamento
    plot_training_loss(training_loss)

    os.makedirs(os.path.dirname(args.modelpath), exist_ok = True)

    # Verificar extensão do arquivo
    if not args.modelpath.endswith('.pth') and not args.modelpath.endswith('.pt'):
        raise ValueError("O caminho do modelo deve ter uma extensão '.pth' ou '.pt'")

    # Salvar modelo treinado
    torch.save(model.state_dict(), args.modelpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de modelo de segmentação de imagens.")
    parser.add_argument("--rgb", type=str, required=True, help="Caminho para o diretório das imagens RGB.")
    parser.add_argument("--groundtruth", type=str, required=True, help="Caminho para o diretório das máscaras de segmentação.")
    parser.add_argument("--modelpath", type=str, required=True, help="Caminho para salvar o modelo treinado.")
    
    args = parser.parse_args()
    main(args)
