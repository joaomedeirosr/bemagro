# Desafio Bem Agro

Pipeline de segmentação de vegetação em ortomosaicos com uma U-Net e encoder ResNet-50.

## Requisitos

Instale as dependências listadas em `requirements.txt`. O treinamento pode ser executado em CPU, mas uma GPU compatível com CUDA é recomendada.

## Organização dos dados

As imagens RGB e suas máscaras devem ficar em diretórios separados e usar o mesmo nome-base:

```text
dataset/
├── images/
│   ├── bloco_0_0.png
│   └── bloco_0_1.png
└── masks/
    ├── bloco_0_0.png
    └── bloco_0_1.png
```

As máscaras devem ser binárias, com valores `0/1` ou `0/255`. Para a validação espacial, os arquivos devem seguir o formato `bloco_X_Y`.

## Treinamento

```bash
python train_model.py \
  --rgb dataset/images \
  --groundtruth dataset/masks \
  --modelpath model/unet_resnet50.pth \
  --epochs 60 \
  --batch-size 2 \
  --device auto
```

O treinamento salva o melhor checkpoint segundo o mIoU de validação. O arquivo inclui a versão da arquitetura, pesos, normalização, métricas e limiar calibrado para inferência.

## Inferência

Para processar uma imagem ou todas as imagens de um diretório:

```bash
python model_inference.py \
  --rgb dataset/images \
  --modelpath model/unet_resnet50.pth \
  --output segmented \
  --device auto
```

O limiar salvo no checkpoint é usado automaticamente. Para substituí-lo:

```bash
python model_inference.py \
  --rgb dataset/images/bloco_0_0.png \
  --modelpath model/unet_resnet50.pth \
  --output segmented/bloco_0_0.png \
  --threshold 0.70
```

> A arquitetura usa um novo decoder com skip connections. Checkpoints antigos, contendo apenas o `state_dict`, não são compatíveis e precisam ser treinados novamente.

## Testes

```bash
python -m unittest discover -s tests -v
```

## Autor

- [Joao Victor Rocha](https://github.com/joaomedeirosr)
