"""
Propósito: Binarizar as imagens RGB cortadas em blocos as etiquetando e criando "mascaras"
Autor : Joao Victor Rocha <jvmedeirosr@gmail.com>
"""
import cv2 as cv
import os
import argparse
import numpy as np

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """ Aplica um filtro gaussiano à imagem para suavização e redução de ruído.

    Args:
        image (np.ndarray): Imagem de entrada em formato numpy array.
        kernel_size (int, opcional): Tamanho do núcleo do filtro gaussiano. Deve ser ímpar. Default é 5.

    Returns:
        np.ndarray: Imagem suavizada com o filtro gaussiano aplicado.
    """
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Aplica um filtro de mediana à imagem para remoção de ruído.

    Args:
        image (np.ndarray): Imagem de entrada em formato numpy array.
        kernel_size (int, opcional): Tamanho do núcleo do filtro de mediana. Deve ser ímpar. Default é 5.

    Returns:
        np.ndarray: Imagem com o filtro de mediana aplicado.
    """
    return cv.medianBlur(image, kernel_size)

def apply_morphological_operations(binary_image: np.ndarray) -> np.ndarray:
    """ Aplica operações morfológicas para limpeza e aprimoramento da imagem binária.

    As operações incluem fechamento para preencher pequenos buracos e abertura para remover pequenas regiões de ruído.

    Args:
        binary_image (np.ndarray): Imagem binária de entrada em formato numpy array.

    Returns:
        np.ndarray: Imagem binária limpa com as operações morfológicas aplicadas.
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # Operação de fechamento para preencher pequenos buracos
    closed_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)
    # Operação de abertura para remover pequenas regiões de ruído
    cleaned_image = cv.morphologyEx(closed_image, cv.MORPH_OPEN, kernel)
    return cleaned_image

def remove_small_components(binary_image: np.ndarray, min_size: int = 500) -> np.ndarray:
    """ Remove pequenos componentes conectados da imagem binária que são menores que o tamanho mínimo especificado.

    Args:
        binary_image (np.ndarray): Imagem binária de entrada em formato numpy array.
        min_size (int, opcional): Tamanho mínimo dos componentes conectados a serem mantidos. Default é 500.

    Returns:
        np.ndarray: Imagem binária com pequenos componentes removidos, mantendo apenas os componentes maiores.
    """
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary_image, connectivity=8)
    # Remove pequenos componentes
    large_components = np.zeros_like(binary_image)
    for i in range(1, num_labels):  # Ignora o fundo
        if stats[i, cv.CC_STAT_AREA] >= min_size:
            large_components[labels == i] = 255
    return large_components

def binarize_image(image_path: str, output_folder: str, threshold: int = 128) -> None:
    """Binariza imagens destacando a vegetação.

    Esta função utiliza um limiar para converter uma imagem em tons de cinza em uma imagem binária.
    Pixels com valor superior ao limiar são considerados como vegetação e recebem o valor 1, enquanto os demais recebem 0.

    Args:
        image_path (str): Caminho para o diretório contendo as imagens em tons de cinza.
        output_folder (str): Caminho para o diretório onde as imagens binarizadas serão salvas.
        threshold (int, optional): Valor do limiar para binarização. Padrão: 128.

    Raises:
        FileNotFoundError: Se o caminho ou imagem nao existir.
        ValueError: Se o limiar for inválido (menor ou igual a 0).
        IOError: Se ocorrer um erro durante o processamento da imagem.
    """

    # Expande o til no caminho do arquivo, se presente
    image_path = os.path.expanduser(image_path)
    output_folder = os.path.expanduser(output_folder)
    
    # Carrega a imagem
    image = cv.imread(image_path)
    
    # Verifica se a imagem foi carregada corretamente
    if image is None:
        print(f"Erro ao carregar o arquivo: {image_path}. Verifique o caminho e a integridade do arquivo.")
        return

    # Converte a imagem para o espaço de cor HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Extraímos os canais de cor
    h, s, v = cv.split(hsv_image)
    
    # Convertemos a imagem para o espaço de cor RGB
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    r, g, b = cv.split(rgb_image)
    
    # Calcula o Índice de Excesso de Verde (ExG)
    exg = (2 * g - r - b)

    # Aplica o filtro Gaussiano para suavizar a imagem
    smoothed_image = apply_gaussian_blur(exg)

    median_blurred_image = apply_median_blur(smoothed_image)

    # Binariza a imagem usando o limiar
    _, binary_image = cv.threshold(median_blurred_image, threshold, 255, cv.THRESH_BINARY)
    
    morphed_image = apply_morphological_operations(binary_image)

    less_noise = remove_small_components(morphed_image)

    final_image = cv.bitwise_not(less_noise)

    # Converte para escala de cinza e salva a imagem binarizada
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv.imwrite(output_path, final_image)

    print(f"Binarização concluída. Imagem binarizada salva em {output_path}")

def main() -> None:
    # Configuração do parser de argumentos
    parser = argparse.ArgumentParser(description="Binarização de blocos de imagem para realçar a assinatura espectral da vegetação.")
    parser.add_argument('--input', required=True, type=str, help='Caminho para o diretório de blocos da imagem')
    parser.add_argument('--output', required=True, type=str, help='Pasta onde as imagens binarizadas serão salvas')
    parser.add_argument('--threshold', default=128, type=int, help='Limiar para binarização (padrão: 128)')

    # Parseia os argumentos
    args = parser.parse_args()
    
    # Cria a pasta de saída, se não existir
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Itera sobre cada imagem no diretório de entrada e realiza a binarização
    for filename in os.listdir(args.input):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(args.input, filename)
            binarize_image(image_path, args.output, args.threshold)

if __name__ == "__main__":
    main()
