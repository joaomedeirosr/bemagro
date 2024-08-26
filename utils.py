"""
Propósito: Criacao de métodos utilitários que ajudam a estimar métricas para o treinamento
Autor : Joao Victor Rocha <jvmedeirosr@gmail.com>
"""
import torch
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calcula o Índice de Jaccard (IoU) entre as previsões e os alvos.

    O Índice de Jaccard é uma métrica para avaliar a sobreposição entre duas imagens binárias.

    Args:
        pred (torch.Tensor): Tensor de previsões do modelo com formato (batch_size, num_classes, height, width).
        target (torch.Tensor): Tensor de alvos com formato (batch_size, height, width), onde os valores são binários.

    Returns:
        float: Valor do Índice de Jaccard (IoU) calculado.
    """
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()
    iou = jaccard_score(target.flatten(), pred.flatten(), average='binary')
    return iou

def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calcula a acurácia das previsões comparadas aos alvos.

    A acurácia é a proporção de previsões corretas em relação ao total de previsões.

    Args:
        pred (torch.Tensor): Tensor de previsões do modelo com formato (batch_size, num_classes, height, width).
        target (torch.Tensor): Tensor de alvos com formato (batch_size, height, width), onde os valores são binários.

    Returns:
        float: Valor da acurácia calculada.
    """
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()

def plot_training_loss(training_loss: list):
    """
    Plota a curva de perda de treinamento ao longo das épocas.

    A função cria um gráfico mostrando como a perda de treinamento evolui durante o treinamento do modelo.

    Args:
        training_loss (list): Lista contendo os valores de perda de treinamento para cada época.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()
