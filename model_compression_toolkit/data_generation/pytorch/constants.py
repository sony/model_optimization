import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_INPUT = 'image_input'
BATCH_AXIS, CHANNEL_AXIS, H_AXIS, W_AXIS = 0, 1, 2, 3
