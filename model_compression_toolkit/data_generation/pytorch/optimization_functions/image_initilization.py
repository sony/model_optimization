# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from functools import partial
from typing import Tuple, Union, List, Callable, Dict

from torch import Tensor
from torchvision.transforms.transforms import _setup_size
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model_compression_toolkit.data_generation.common.enums import DataInitType
from model_compression_toolkit.data_generation.common.constants import NUM_INPUT_CHANNELS


class RandomDataset(Dataset):
    """
    A PyTorch dataset that generates random Gaussian samples with mean 0 and variance 1.
    """

    def __init__(self,
                 length: int,
                 sample_fn: Callable,
                 size: Union[Tuple, List],
                 ):
        """
        Initialize the RandomDataset.

        Args:
            length (int): The number of samples in the dataset.
            sample_fn (Callable): The function to generate a random sample.
            size (Tuple or List): The size of each sample.
        """
        self.length = length
        self.sample_fn = sample_fn
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
        return self.sample_fn(self.size)

def gaussian_sample(size: Tuple[int, ...]) -> Tensor:
    """
    Generate a random Gaussian sample with mean 0 and variance 1.

    Args:
        size (Tuple[int, ...]): The size of the sample.

    Returns:
        torch.Tensor: The random Gaussian sample.
    """
    return torch.randn(size).float()

def diverse_sample(size: Tuple[int, ...]) -> Tensor:
    """
    Generate a random diverse sample with random mean and standard deviation.

    Args:
        size (Tuple[int, ...]): The size of the sample.

    Returns:
        torch.Tensor: The random diverse sample.
    """
    random_std = torch.randn(size=(NUM_INPUT_CHANNELS, 1, 1))
    random_mean = torch.randn(size=(NUM_INPUT_CHANNELS, 1, 1))

    sample = random_std * torch.randn(size) + random_mean

    # filtering to make the image a bit smoother
    kernel = torch.ones(NUM_INPUT_CHANNELS, NUM_INPUT_CHANNELS, 5, 5) / 16
    sample = F.conv2d(sample, kernel, padding=1)
    return sample.float()

def default_data_init_fn(
        n_images: int = 1000,
        size: Union[int, Tuple[int, int]] = (224, 224),
        crop: int = 32,
        sample_fn: Callable = diverse_sample,
        batch_size: int = 50) -> Tuple[int, DataLoader]:
    """
    Get a DataLoader with random samples.

    Args:
        n_images (int): The number of random samples.
        size (Union[int, Tuple[int, int]]): The size of each sample.
        crop (int): The crop size.
        sample_fn (Callable): The function to generate a random sample.
        batch_size (int): The batch size.

    Returns:
        Tuple[int, DataLoader]: A tuple containing the length of the DataLoader and the DataLoader object.
    """
    image_size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
    image_size = [s + crop for s in image_size]
    dataset = RandomDataset(length=n_images, size=[NUM_INPUT_CHANNELS] + image_size,
                            sample_fn=sample_fn)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
    return data_loader


# Dictionary of image initialization functions
image_initialization_function_dict: Dict[DataInitType, Callable] = {
    DataInitType.Gaussian: partial(default_data_init_fn, sample_fn=gaussian_sample),
    DataInitType.Diverse: partial(default_data_init_fn, sample_fn=diverse_sample),
}