from typing import Tuple, Union, List

import cv2
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, CenterCrop
from torchvision.transforms.transforms import _setup_size
from model_compression_toolkit.data_generation.common.data_generation_config import BaseImagePipeline
import torch
import numpy as np


class RandomDataset(Dataset):
    """
    A PyTorch dataset that generates random Gaussian samples with mean 0 and variance 1.
    """

    def __init__(self,
                 length: int,
                 size: Union[Tuple, List],
                 ):
        """
        Initialize the RandomDataset.

        Args:
            length (int): The number of samples in the dataset.
            size (tuple or list): The size of each sample.
        """
        self.length = length
        self.size = size

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.length

    def __getitem__(self, idx: int) -> Tensor:
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            Tensor: The random sample.
        """
        random_std = torch.from_numpy(np.array([1, 1, 1]))[:, None, None] * (torch.randn(size=(3, 1, 1))) + torch.ones(size=(3, 1, 1))
        random_mean = torch.from_numpy(np.array([1, 1, 1]))[:, None, None] * torch.randn(size=(3, 1, 1)) * torch.ones(self.size)

        sample = random_std * torch.randn(self.size) + random_mean
        kernel = np.ones((5, 5), np.float32) / 16
        sample = torch.from_numpy(cv2.filter2D(sample.float().detach().cpu().numpy(), -1, kernel))
        return sample.float()


def get_random_data(
        n_images: int = 1000,
        size: tuple = (224, 224),
        crop: int = 32,
        batch_size: int = 50) -> Tuple[int, Tensor]:
    """
    Get a random sample DataLoader.

    Args:
        n_images (int): The number of random samples.
        size (tuple): The size of each sample.
        crop (int): The crop size.
        batch_size (int): The batch size.

    Returns:
        tuple: A tuple containing the length of the DataLoader and the DataLoader object.
    """
    image_size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
    image_size = [s + crop for s in image_size]
    dataset = RandomDataset(length=n_images, size=[3] + image_size)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
    return len(data_loader), data_loader


class PytorchImagePipeline(BaseImagePipeline):
    """
    An image pipeline implementation for PyTorch models.
    """
    def __init__(self, output_image_size: int, padding: int = 0):
        """
        Initialize the PytorchImagePipeline.

        Args:
            output_image_size (int): The output image size.
            padding (int): The padding size.
        """
        super(PytorchImagePipeline, self).__init__(output_image_size, padding)
        self.random_crop = RandomCrop(self.output_image_size)
        self.random_flip = RandomHorizontalFlip(0.5)
        self.center_crop = CenterCrop(self.output_image_size)

    def get_image_input_size(self) -> int:
        """
        Get the input size of the image.

        Returns:
            int: The input image size.
        """
        return self.output_image_size + self.padding

    def image_input_manipulation(self, images: Tensor) -> Tensor:
        """
        Manipulate the input images.

        Args:
            images (Tensor): The input images.

        Returns:
            Tensor: The manipulated images.
        """
        random_flipped_data = self.random_flip(images)
        return self.random_crop(random_flipped_data)

    def image_output_finalize(self, images: Tensor) -> Tensor:
        """
        Finalize the output images.

        Args:
            images (Tensor): The output images.

        Returns:
            Tensor: The finalized images.
        """
        return self.center_crop(images)