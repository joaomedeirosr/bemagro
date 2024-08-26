"""
Propósito: Dividir a imagem orthomosaic.tif em pequenos blocos 512x512 e generar o dataset
Autor : Joao Victor Rocha <jvmedeirosr@gmail.com>
"""
import cv2 as cv
import os
import argparse
import numpy as np
import albumentations as A  

def remove_black_white_borders(image: np.ndarray) -> np.ndarray:
    """Funcao com objetivo de remover bordas brancas e
    pretas de uma imagem.

    Args:
    - image (np.ndarray): A imagem carregada em BGR

    Returns:
    - cropped_image: A imagem cortada sem bordas brancas e pretas.

    NOTE: O type, numpy.ndarray é uma maneira mas antiga porém mais simplificada
    caso seja necessário modernizar, utilizar o type NDArray[np.uint8], importanto o módulo:
    from numpy.typing import NDArray
    """
    
    # Converter a imagem para escala de cinza
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Criar uma máscara onde os pixels úteis (não brancos e não pretos) são 1, e os brancos ou pretos são 0
    mask = cv.inRange(gray, 1, 254)
    
    # Encontrar todas as linhas e colunas que possuem apenas pixels úteis
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Determinar as coordenadas das linhas e colunas que limitam a área útil
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Recortar a imagem nas coordenadas determinadas
    cropped_image = image[y_min + 10 : y_max - 10, x_min + 10 : x_max - 10]
    
    return cropped_image


def split_image(image: np.ndarray, output_folder: str, block_size: int, num_augmentations: int = 15) -> None:
    """ Funcao com objetivo de cortar a imagem orthomosaic.tif em 
        blocos iguais de 512 x 512 px.
    
    Args: 
       - image (np.ndarray): A imagem carregada 
       - output_folder (str): Caminho da pasta onde as imagens serao salvas. 
       - block_size (int): Tamanho dos blocos a serem cortados.
    """    

    height, width, _ = image.shape

    print(f"Dimensões da imagem carregada: ({width}x{height}) px")
    
    # Calcula o número de blocos que cabem na imagem original
    x_blocks = width // block_size
    y_blocks = height// block_size
    print(f"Blocos horizontais: {x_blocks} , Blocos verticais: {y_blocks}")
    
    # Definindo transformações de data augmentation usando Albumentations
    all_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.Blur(blur_limit=3),
        A.ElasticTransform(alpha=1.0, sigma=50.0),
        A.Affine(rotate=(-45, 45), p=0.5),
        A.RandomCrop(width=block_size, height=block_size, p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.PadIfNeeded(min_height=block_size, min_width=block_size, border_mode=cv.BORDER_CONSTANT, value=(0,0,0), p=1.0),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    ])

    # Selecionar aleatoriamente o número de augmentations desejado
    if num_augmentations > len(all_transforms):
        num_augmentations = len(all_transforms)
    
    selected_transforms = all_transforms[:num_augmentations]
    transform = A.Compose(selected_transforms)

    # Itera sobre cada bloco e salva como PNG
    for i in range(x_blocks):
        for j in range(y_blocks):
            x_start = i * block_size
            y_start = j * block_size

            x_end = x_start + block_size # Ajusta o bloco final para não exceder a largura da imagem
            y_end = y_start + block_size  # Ajusta o bloco final para não exceder a altura da imagem

            # Recorta a imagem gerando os blocos
            cropped_image = image[y_start : y_end, x_start : x_end]

            # Redimensiona o bloco para o tamanho exato
            cropped_image = cv.resize(cropped_image, (block_size, block_size), interpolation=cv.INTER_AREA)
            
            # Define o caminho de saída
            output_path = os.path.join(output_folder, f'bloco_{i}_{j}.png')
            
            # Salva o bloco como PNG
            cv.imwrite(output_path, cropped_image)

            for k in range(num_augmentations):
                # Seleciona uma transformação aleatória para aplicar
                transform = A.Compose([all_transforms[k % len(all_transforms)]])
                augmented = transform(image=cropped_image)["image"]
                # Nome do arquivo para cada augmentação aplicada
                augmented_output_path = os.path.join(output_folder, f'bloco_aug_{i}_{j}_{k}.png')
                cv.imwrite(augmented_output_path, augmented)

    print(f"Divisão concluída. Blocos salvos na pasta {output_folder}")

def main() -> None:

    # Configuração do parser de argumentos
    parser = argparse.ArgumentParser(description="Divida um ortomosaico TIFF em blocos menores.")
    parser.add_argument('--input', required = True, type=str, help='Caminho para o arquivo ortomosaico .tif')
    parser.add_argument('--output', required = True, type=str, help='Pasta onde os blocos serão salvos')
    parser.add_argument('--block-size', default = 512, type=int, help='Tamanho dos blocos (padrão: 512x512)')
    parser.add_argument('--num-augmentations', default=15, type=int, help='Número de augmentations a serem aplicadas por imagem')
    # Parseia os argumentos
    args = parser.parse_args()

# Cria a pasta de saída, se não existir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Le a imagem original orthomosaic.tif original
    image = cv.imread(args.input)

    # Verifica se a imagem foi carregada com sucesso
    if image is None:
        print("Erro ao carregar o ortomosaico")
        return
    
    # Remover imperfeicoes da imagem
    cropped_image = remove_black_white_borders(image)

    # Chama a função principal para dividir a imagem
    split_image(cropped_image,args.output, args.block_size,args.num_augmentations)

if __name__ == "__main__":
    main()
