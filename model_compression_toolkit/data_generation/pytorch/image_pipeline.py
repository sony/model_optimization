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
from typing import Type, Dict, Tuple, Union, List

from torch import Tensor
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, CenterCrop

from model_compression_toolkit.data_generation.common.enums import ImagePipelineType
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline
from model_compression_toolkit.data_generation.pytorch.image_operations import Smoothing, create_valid_grid


class PytorchIdentityImagePipeline(BaseImagePipeline):
    """
    An image pipeline implementation for PyTorch models that returns the input images as is (identity).
    """
    def __init__(self,
                 output_image_size: Union[int, Tuple[int, int]],
                 extra_pixels: Union[int, Tuple[int, int]] = 0,
                 normalization: List[List[int]] = [[0, 0, 0], [1, 1, 1]],
                 image_clipping: bool = True,
                 ):
        """
        Initialize the PytorchIdentityImagePipeline.

        Args:
            output_image_size (Union[int, Tuple[int, int]]): The output image size.
            extra_pixels (Union[int, Tuple[int, int]]): Extra pixels to add to the input image size (not used in identity pipeline).
            normalization (List[List[float]]): The image normalization values for processing images during optimization.
            image_clipping (bool): Whether to clip images during optimization.
        """
        super(PytorchIdentityImagePipeline, self).__init__(output_image_size, extra_pixels, image_clipping, normalization)

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
                 normalization: List[List[int]] = [[0, 0, 0], [1, 1, 1]],
                 image_clipping: bool = True,
                 smoothing_filter_size: int = 3,
                 smoothing_filter_sigma: float = 1.25):
        """
        Initialize the PytorchRandomCropFlipImagePipeline.

        Args:
            output_image_size (Union[int, Tuple[int, int]]): The output image size.
            extra_pixels (Union[int, Tuple[int, int]]): Extra pixels to add to the input image size. Defaults to 0.
            normalization (List[List[float]]): The image normalization values for processing images during optimization.
            image_clipping (bool): Whether to clip images during optimization.
            smoothing_filter_size (int): The size of the smoothing filter. Defaults to 3.
            smoothing_filter_sigma (float): The standard deviation of the smoothing filter. Defaults to 1.25.
        """
        super(PytorchSmoothAugmentationImagePipeline, self).__init__(output_image_size, extra_pixels, image_clipping, normalization)
        self.smoothing = Smoothing(size=smoothing_filter_size, sigma=smoothing_filter_sigma)
        self.random_crop = RandomCrop(self.output_image_size)
        self.random_flip = RandomHorizontalFlip(0.5)
        self.center_crop = CenterCrop(self.output_image_size)
        self.valid_grid = create_valid_grid(means=self.normalization[0], stds=self.normalization[1])

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
        if self.image_clipping:
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
        if self.image_clipping:
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