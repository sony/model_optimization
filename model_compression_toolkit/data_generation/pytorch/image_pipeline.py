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
from typing import Type, Dict, List

from torch import Tensor
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, CenterCrop


from model_compression_toolkit.data_generation.common.enums import ImagePipelineType, ImageNormalizationType
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline


class PytorchIdentityImagePipeline(BaseImagePipeline):
    """
    An image pipeline implementation for PyTorch models that returns the input images as is (identity).
    """
    def __init__(self, output_image_size: int, extra_pixels: int = 0):
        """
        Initialize the PytorchIdentityImagePipeline.

        Args:
            output_image_size (int): The output image size.
            extra_pixels (int): Extra pixels to add to the input image size (not used in identity pipeline).
        """
        super(PytorchIdentityImagePipeline, self).__init__(output_image_size)

    def get_image_input_size(self) -> int:
        """
        Get the input size of the image.

        Returns:
            int: The input image size.
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


class PytorchRandomCropImagePipeline(BaseImagePipeline):
    """
    An image pipeline implementation for PyTorch models that includes random cropping.
    """
    def __init__(self, output_image_size: int, extra_pixels: int = 0):
        """
        Initialize the PytorchRandomCropFlipImagePipeline.

        Args:
            output_image_size (int): The output image size.
            extra_pixels (int): Extra pixels to add to the input image size. Defaults to 0.
        """
        super(PytorchRandomCropImagePipeline, self).__init__(output_image_size)
        self.extra_pixels = extra_pixels
        self.random_crop = RandomCrop(self.output_image_size)
        self.center_crop = CenterCrop(self.output_image_size)

    def get_image_input_size(self) -> int:
        """
        Get the input size of the image.

        Returns:
            int: The input image size.
        """
        return self.output_image_size + self.extra_pixels

    def image_input_manipulation(self, images: Tensor) -> Tensor:
        """
        Manipulate the input images with random flipping and cropping.

        Args:
            images (Tensor): The input images.

        Returns:
            Tensor: The manipulated images (randomly flipped and cropped).
        """
        return self.random_crop(images)

    def image_output_finalize(self, images: Tensor) -> Tensor:
        """
        Finalize the output images with center cropping.

        Args:
            images (Tensor): The output images.

        Returns:
            Tensor: The finalized images (center cropped).
        """
        return self.center_crop(images)


class PytorchRandomCropFlipImagePipeline(BaseImagePipeline):
    """
    An image pipeline implementation for PyTorch models that includes random cropping and flipping.
    """
    def __init__(self, output_image_size: int, extra_pixels: int = 0):
        """
        Initialize the PytorchRandomCropFlipImagePipeline.

        Args:
            output_image_size (int): The output image size.
            extra_pixels (int): Extra pixels to add to the input image size. Defaults to 0.
        """
        super(PytorchRandomCropFlipImagePipeline, self).__init__(output_image_size)
        self.extra_pixels = extra_pixels
        self.random_crop = RandomCrop(self.output_image_size)
        self.random_flip = RandomHorizontalFlip(0.5)
        self.center_crop = CenterCrop(self.output_image_size)

    def get_image_input_size(self) -> int:
        """
        Get the input size of the image.

        Returns:
            int: The input image size.
        """
        return self.output_image_size + self.extra_pixels

    def image_input_manipulation(self, images: Tensor) -> Tensor:
        """
        Manipulate the input images with random flipping and cropping.

        Args:
            images (Tensor): The input images.

        Returns:
            Tensor: The manipulated images (randomly flipped and cropped).
        """
        random_flipped_data = self.random_flip(images)
        return self.random_crop(random_flipped_data)

    def image_output_finalize(self, images: Tensor) -> Tensor:
        """
        Finalize the output images with center cropping.

        Args:
            images (Tensor): The output images.

        Returns:
            Tensor: The finalized images (center cropped).
        """
        return self.center_crop(images)


# Dictionary mapping ImagePipelineType to corresponding image pipeline classes
image_pipeline_dict: Dict[ImagePipelineType, Type[BaseImagePipeline]] = {
    ImagePipelineType.IDENTITY: PytorchIdentityImagePipeline,
    ImagePipelineType.RANDOM_CROP: PytorchRandomCropImagePipeline,
    ImagePipelineType.RANDOM_CROP_FLIP: PytorchRandomCropFlipImagePipeline
}

# Dictionary mapping ImageNormalizationType to corresponding normalization values
image_normalization_dict: Dict[ImageNormalizationType, List[List[float]]] = {
    ImageNormalizationType.TORCHVISION: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    ImageNormalizationType.NO_NORMALIZATION: [[0, 0, 0], [1, 1, 1]]
}