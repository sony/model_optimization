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
import numpy as np
import torch
import torch.nn.functional as F
from typing import Type, Dict, List, Tuple, Union

from torch import Tensor
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, Normalize

from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.data_generation.common.enums import ImagePipelineType, ImageNormalizationType
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline

class Smoothing(torch.nn.Module):
    """
    A PyTorch module for applying Gaussian smoothing to an image.
    """

    def __init__(self, size: int = 3, sigma: float = 1.25, kernel: torch.Tensor = None):
        """
        Initialize the Smoothing module.

        Args:
            size (int): The size of the Gaussian kernel.
            sigma (float): The standard deviation of the Gaussian kernel.
            kernel (torch.Tensor, optional): Precomputed Gaussian kernel. If None, it will be created.
        """
        super().__init__()
        if kernel is None:
            kernel = self.gaussian_kernel(size, sigma)
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
        return f"{self.__class__.__name__}(kernel={self.kernel.shape[-1]})"

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


class PytorchIdentityImagePipeline(BaseImagePipeline):
    """
    An image pipeline implementation for PyTorch models that returns the input images as is (identity).
    """
    def __init__(self, output_image_size: Union[int, Tuple[int, int]], extra_pixels: Union[int, Tuple[int, int]] = 0):
        """
        Initialize the PytorchIdentityImagePipeline.

        Args:
            output_image_size (Union[int, Tuple[int, int]]): The output image size.
            extra_pixels (Union[int, Tuple[int, int]]): Extra pixels to add to the input image size (not used in identity pipeline).
        """
        super(PytorchIdentityImagePipeline, self).__init__(output_image_size, extra_pixels)

    def get_image_input_size(self) -> Tuple[int, int]:
        """
        Get the input size of the image.

        Returns:
            Tuple[int, int]: The input image size.
        """
        return self.output_image_size

    def image_input_manipulation(self, images: Tensor) -> Tensor:
        """
        Manipulate the input images (identity operation, returns the input images as is).

        Args:
            images (Tensor): The input images.

        Returns:
            Tensor: The manipulated images (input images as is).
        """
        return images

    def image_output_finalize(self, images: Tensor) -> Tensor:
        """
        Finalize the output images (identity operation, returns the output images as is).

        Args:
            images (Tensor): The output images.

        Returns:
            Tensor: The finalized images (output images as is).
        """
        return images


class PytorchSmoothAugmentationImagePipeline(BaseImagePipeline):
    """
    An image pipeline implementation for PyTorch models that includes random cropping and flipping.
    """
    def __init__(self,
                 output_image_size: Union[int, Tuple[int, int]],
                 extra_pixels: Union[int, Tuple[int, int]] = 0,
                 smoothing_filter_size: int = 3,
                 smoothing_filter_sigma: float = 1.25):
        """
        Initialize the PytorchRandomCropFlipImagePipeline.

        Args:
            output_image_size (Union[int, Tuple[int, int]]): The output image size.
            extra_pixels (Union[int, Tuple[int, int]]): Extra pixels to add to the input image size. Defaults to 0.
            smoothing_filter_size (int): The size of the smoothing filter. Defaults to 3.
            smoothing_filter_sigma (float): The standard deviation of the smoothing filter. Defaults to 1.25.
        """
        super(PytorchSmoothAugmentationImagePipeline, self).__init__(output_image_size, extra_pixels)
        self.smoothing = Smoothing(size=smoothing_filter_size, sigma=smoothing_filter_sigma)
        self.random_crop = RandomCrop(self.output_image_size)
        self.random_flip = RandomHorizontalFlip(0.5)
        self.center_crop = CenterCrop(self.output_image_size)

        # Image valid grid
        pixel_grid = torch.from_numpy(np.array(list(range(256))).repeat(3).reshape(-1, 3) / 255)
        self.valid_grid = Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(pixel_grid.transpose(1, 0)[None, :, :, None]).squeeze().to(get_working_device())

    def get_image_input_size(self) -> Tuple[int, int]:
        """
        Get the input size of the image.

        Returns:
            Tuple[int, int]: The input image size.
        """
        return tuple([o + e for (o, e) in zip(self.output_image_size, self.extra_pixels)])

    def image_input_manipulation(self, images: Tensor) -> Tensor:
        """
        Manipulate the input images with random flipping and cropping.

        Args:
            images (Tensor): The input images.

        Returns:
            Tensor: The manipulated images.
        """
        new_images = self.random_flip(images)
        new_images = self.smoothing(new_images)
        new_images = self.random_crop(new_images)
        new_images = self.clip_images(new_images, self.valid_grid)
        return new_images

    def image_output_finalize(self, images: Tensor) -> Tensor:
        """
        Finalize the output images with center cropping.

        Args:
            images (Tensor): The output images.

        Returns:
            Tensor: The finalized images (center cropped).
        """
        new_images = self.smoothing(images)
        new_images = self.center_crop(new_images)
        new_images = self.clip_images(new_images, self.valid_grid)
        return new_images

    @staticmethod
    def clip_images(images: Tensor, valid_grid: Tensor, reflection: bool = False) -> Tensor:
        """
        Clip the images based on a valid grid.

        Args:
            images (Tensor): The images to be clipped.
            valid_grid (Tensor): The valid grid for clipping.
            reflection (bool): Whether to apply reflection during clipping. Defaults to False.

        Returns:
            Tensor: The clipped images.
        """
        with torch.no_grad():
            for i_ch in range(valid_grid.shape[0]):
                clamp = torch.clamp(images[:, i_ch, :, :], valid_grid[i_ch, :].min(), valid_grid[i_ch, :].max())
                if reflection:
                    images[:, i_ch, :, :] = 2 * clamp - images[:, i_ch, :, :]
                else:
                    images[:, i_ch, :, :] = clamp
        return images


# Dictionary mapping ImagePipelineType to corresponding image pipeline classes
image_pipeline_dict: Dict[ImagePipelineType, Type[BaseImagePipeline]] = {
    ImagePipelineType.IDENTITY: PytorchIdentityImagePipeline,
    ImagePipelineType.SMOOTHING_AND_AUGMENTATION: PytorchSmoothAugmentationImagePipeline
}

# Dictionary mapping ImageNormalizationType to corresponding normalization values
image_normalization_dict: Dict[ImageNormalizationType, List[List[float]]] = {
    ImageNormalizationType.TORCHVISION: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    ImageNormalizationType.NO_NORMALIZATION: [[0, 0, 0], [1, 1, 1]]
}