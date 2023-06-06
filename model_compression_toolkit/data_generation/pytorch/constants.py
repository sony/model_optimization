import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_INPUT = 'image_input'
