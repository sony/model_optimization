import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, CenterCrop
from model_compression_toolkit.data_generation.common.data_generation_config import BaseImagePipeline
import torch
from model_compression_toolkit.data_generation.pytorch.constants import DEVICE
import numpy as np


class RandomDataset(Dataset):
    """
    get random gaussian samples with mean 0 and variance 1
    """

    def __init__(self,
                 length,
                 size,
                 ):
        self.length = length
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        random_std = torch.from_numpy(np.array([1, 1, 1]))[:, None, None] * (torch.randn(size=(3, 1, 1))) + torch.ones(size=(3, 1, 1))
        random_mean = torch.from_numpy(np.array([1, 1, 1]))[:, None, None] * torch.randn(size=(3, 1, 1)) * torch.ones(self.size)

        sample = random_std * torch.randn(self.size) + random_mean
        kernel = np.ones((5, 5), np.float32) / 16
        sample = torch.from_numpy(cv2.filter2D(sample.float().detach().cpu().numpy(), -1, kernel))
        return sample.float()


def get_random_data(
        n_images=1000,
        image_size: int = 224,
        batch_size=50):
    """
    get random sample dataloader
    dataset: name of the dataset
    batch_size: the batch size of random data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    dataset = RandomDataset(length=n_images, size=(3, image_size, image_size))
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)
    return len(data_loader), data_loader


class PytorchImagePipeline(BaseImagePipeline):
    def get_image_input_size(self):
        return self.output_image_size

    def image_input_manipulation(self, images):
        random_crop = RandomCrop(224)
        random_flip = RandomHorizontalFlip(0.5)
        distill_roll = 4
        roll1 = torch.randint(0, distill_roll + 1, size=(1,)).item() * np.random.choice([-1, 1])
        roll2 = torch.randint(0, distill_roll + 1, size=(1,)).item() * np.random.choice([-1, 1])
        rolled_data = torch.roll(images, shifts=(roll1, roll2), dims=(2, 3))
        random_flipped_data = random_flip(rolled_data).to(DEVICE)
        return random_crop(random_flipped_data).to(DEVICE)

    def image_output_finalize(self, images):
        center_crop = CenterCrop(self.output_image_size)
        return center_crop(images)