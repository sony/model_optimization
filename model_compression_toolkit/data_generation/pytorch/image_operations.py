# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from torchvision.transforms import Normalize

from model_compression_toolkit.logger import Logger


def create_valid_grid(means: List[int], stds: List[int]) -> torch.Tensor:
    """
    Create a valid grid for image normalization.

    Args:
        means (List[int]): List of mean values per channel.
        stds (List[int]): List of standard deviation values per channel.

    Returns:
        torch.Tensor: The valid grid for image normalization.
    """
    # Image valid grid
    pixel_grid = torch.from_numpy(np.array(list(range(256))).repeat(3).reshape(-1, 3)).float()
    valid_grid = Normalize(mean=means, std=stds)(pixel_grid.transpose(1, 0)[None, :, :, None]).squeeze().to(
        get_working_device())
    return valid_grid


class Smoothing(torch.nn.Module):
    """
    A PyTorch module for applying Gaussian smoothing to an image.
    """

    def __init__(self, size: int = 3, sigma: float = 1.25, kernel: torch.Tensor = None):
        """
        Initialize the Smoothing module.

        Args:
            size (int): The size of the Gaussian kernel (Default: 3).
            sigma (float): The standard deviation of the Gaussian kernel (Defalut: 1.25).
            kernel (torch.Tensor, optional): Precomputed Gaussian kernel. If None, it will be created.
        """
        super().__init__()
        if kernel is None:
            kernel = self.gaussian_kernel(size, sigma)
        if kernel.dim() != 2:
            Logger.critical("Kernel must have 2 dimensions. Found {} dimensions.".format(kernel.dim())) # pragma: no cover
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        # Repeat for 3 color channels
        kernel = kernel.repeat(3, 1, 1, 1)
        self.kernel = kernel

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian smoothing to the input image.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The smoothed image tensor.
        """
        return F.conv2d(image, self.kernel.to(image.device), padding=self.kernel.shape[-1] // 2, groups=3)

    def __repr__(self) -> str:
        """
        Return the string representation of the Smoothing module.

        Returns:
            str: String representation of the Smoothing module.
        """
        return f"{self.__class__.__name__}(kernel={self.kernel.shape[-1]})" # pragma: no cover

    @staticmethod
    def gaussian_kernel(size: int = 3, sigma: float = 1) -> torch.Tensor:
        """
        Create a Gaussian kernel.

        Args:
            size (int): The size of the Gaussian kernel.
            sigma (float): The standard deviation of the Gaussian kernel.

        Returns:
            torch.Tensor: The Gaussian kernel tensor.
        """
        axis = torch.arange(-size // 2 + 1., size // 2 + 1.)
        x, y = torch.meshgrid(axis, axis)
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / torch.sum(kernel)
        return kernel
