"""
Propósito: Realizar a inferência do modelo
Autor : Joao Victor Rocha <jvmedeirosr@gmail.com>
"""
import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
from model import UNetResNet


def load_model(model_path, device):
    """
    Carrega um modelo treinado a partir de um arquivo e o transfere para o dispositivo especificado.

    Args:
        model_path (str): Caminho para o arquivo do modelo treinado.
        device (torch.device): Dispositivo (CPU ou GPU) para o qual o modelo será transferido.

    Returns:
        torch.nn.Module: Modelo carregado e preparado para inferência.
    """
    model = UNetResNet(n_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

def process_single_image(model, file_path, output_path, device):
    """
    Processa uma única imagem, realiza inferência utilizando o modelo e salva a imagem segmentada resultante.

    Args:
        model (torch.nn.Module): Modelo treinado para realizar a inferência.
        file_path (str): Caminho para o arquivo da imagem de entrada.
        output_path (str): Caminho para salvar a imagem segmentada resultante.
        device (torch.device): Dispositivo (CPU ou GPU) para realizar a inferência.

    Returns:
        None
    """
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização
    ])

    # Carregar imagem e aplicar transformações
    try:
        image = Image.open(file_path).convert('RGB')
    except (IOError, UnidentifiedImageError):
        print(f"Arquivo não é uma imagem válida: {file_path}. Ignorando...")
        return
    
    original_image = image.copy()
    image = transform(image).unsqueeze(0).to(device)

    # Realizar inferência
    with torch.no_grad():
        output = model(image)
        # Aplicar uma função de ativação para obter a máscara binária
        output = torch.softmax(output, dim=1)  # Se sua saída tem mais de um canal.
    mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

  

    # Ajuste na geração da imagem binária
    if mask.ndim == 3:
        mask = mask[0]  # Remover a dimensão extra se necessário

    output_image = Image.fromarray((mask * 255).astype(np.uint8))  # Multiplicar por 255 para obter a escala de cinza

    # Criar diretório de saída se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_image.save(output_path)
    print(f"Imagem segmentada salva em: {output_path}")

def segment_image(model, image_path, output_dir, device):
    """
    Segmenta todas as imagens em um diretório ou processa uma única imagem, salvando os resultados no diretório de saída.

    Args:
        model (torch.nn.Module): Modelo treinado para realizar a segmentação.
        image_path (str): Caminho para o diretório contendo imagens ou para uma única imagem.
        output_dir (str): Diretório onde as imagens segmentadas serão salvas.
        device (torch.device): Dispositivo (CPU ou GPU) para realizar a inferência.

    Returns:
        None
    """
    # Verificar se o caminho da imagem é um diretório
    if os.path.isdir(image_path):
        # Iterar por todas as imagens no diretório
        for file_name in os.listdir(image_path):
            file_path = os.path.join(image_path, file_name)
            if os.path.isfile(file_path):
                # Verificar se o arquivo é uma imagem suportada
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    output_path = os.path.join(output_dir, f"seg_{file_name}")
                    process_single_image(model, file_path, output_path, device)
                else:
                    print(f"Arquivo ignorado (não é uma imagem suportada): {file_name}")
    else:
        # Processar uma única imagem
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            output_path = output_dir
            process_single_image(model, image_path, output_path, device)
        else:
            print(f"O arquivo especificado não é uma imagem suportada: {image_path}")

def main():
    parser = argparse.ArgumentParser(description="Inferência de modelo de segmentação de vegetação.")
    parser.add_argument("--rgb", type=str, required=True, help="Caminho para a imagem RGB ou diretório de imagens.")
    parser.add_argument("--modelpath", type=str, required=True, help="Caminho para o modelo treinado (.pth ou .pt).")
    parser.add_argument("--output", type=str, required=True, help="Caminho para salvar a imagem segmentada ou diretório de saída para várias imagens.")
    
    args = parser.parse_args()

     # Criar diretório de saída se não existir
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Criando diretório de saída: {args.output}")

    # Configurações de dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carregar modelo
    model = load_model(args.modelpath, device)

    # Realizar segmentação na(s) imagem(ns) de entrada
    segment_image(model, args.rgb, args.output, device)

if __name__ == "__main__":
    main()
