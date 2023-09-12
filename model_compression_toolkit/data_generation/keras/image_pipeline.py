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
from typing import Tuple, Dict, Type, List

import numpy as np
import tensorflow as tf

from model_compression_toolkit.data_generation.common.enums import ImagePipelineType, ImageNormalizationType
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline


# Define tf function for image manipulation

def random_crop(image: tf.Tensor,
                height_crop: int,
                width_crop: int) -> tf.Tensor:
    """
    Randomly crop an image to the specified size.

    Args:
        image (tf.Tensor): Input image tensor.
        height_crop (int): Size of the crop in the height axis.
        width_crop (int): Size of the crop in the width axis.

    Returns:
        tf.Tensor: Cropped image tensor.
    """
    cropped_image = tf.image.random_crop(image,
                                         size=(tf.shape(image)[0],
                                               height_crop,
                                               width_crop,
                                               tf.shape(image)[-1]))
    return cropped_image


def center_crop(image: tf.Tensor,
                output_size: Tuple) -> tf.Tensor:
    """
    Center crop an image to the specified size.

    Args:
        image (tf.Tensor): Input image tensor.
        output_size (Tuple): Size of image after the crop (height and width).

    Returns:
        tf.Tensor: Cropped image tensor.
    """

    # Calculate the cropping dimensions
    input_shape = tf.shape(image)
    height, width = input_shape[1], input_shape[2]
    target_height, target_width = output_size[0], output_size[1]

    # Calculate the cropping offsets
    offset_height = tf.maximum((height - target_height) // 2, 0)
    offset_width = tf.maximum((width - target_width) // 2, 0)

    # Crop the image
    cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

    return cropped_image


def random_flip(image: tf.Tensor) -> tf.Tensor:
    """
    Randomly flip an image horizontally with a specified probability.

    Args:
        image (tf.Tensor): Input image tensor.

    Returns:
        tf.Tensor: Flipped image tensor.
    """
    flip_image = tf.image.random_flip_left_right(image)
    return flip_image


class TensorflowCropFlipImagePipeline(BaseImagePipeline):
    def __init__(self,
                 output_image_size: Tuple,
                 extra_pixels: int):
        """
        Initialize the TensorflowCropFlipImagePipeline.

        Args:
            output_image_size (Tuple): The output image size.
            extra_pixels (int): Extra pixels to add to the input image size. Defaults to 0.
        """
        super(TensorflowCropFlipImagePipeline, self, ).__init__(output_image_size, extra_pixels)

        # List of image manipulation functions and their arguments.
        self.img_manipulation_list = [(random_flip, {}),
                                      (random_crop, {'height_crop': output_image_size[0],
                                                     'width_crop': output_image_size[1]})]

        # List of output image manipulation functions and their arguments.
        self.img_output_finalize_list = [(center_crop, {'output_size': output_image_size})]
        self.extra_pixels = extra_pixels

    def get_image_input_size(self) -> Tuple:
        """
        Get the size of the input image considering extra pixels.

        Returns:
            Tuple: Size of the input image.
        """
        return tuple(np.array(self.output_image_size) + self.extra_pixels)

    def image_input_manipulation(self,
                                 images: tf.Tensor) -> tf.Tensor:
        """
        Apply image manipulation functions to input images.

        Args:
            images (tf.Tensor): Input images.

        Returns:
            tf.Tensor: Manipulated images.
        """

        def manipulate_fn(image):
            for fn, args in self.img_manipulation_list:
                image = fn(image, **args)
            return image

        manipulated_images = manipulate_fn(images)
        return manipulated_images

    def image_output_finalize(self,
                              images: tf.Tensor) -> tf.Tensor:
        """
        Apply image output finalization functions to images.

        Args:
            images (tf.Tensor): Input images.

        Returns:
            tf.Tensor: Finalized images.
        """

        def finalize_fn(image):
            for fn, args in self.img_output_finalize_list:
                image = fn(image, **args)
            return image

        output_finalize_images = finalize_fn(images)
        return output_finalize_images


class TensorflowIdentityImagePipeline(BaseImagePipeline):

    def __init__(self, output_image_size: int,
                 extra_pixels: int
                 ):
        """
        Initialize the TensorflowIdentityImagePipeline.

        Args:
            output_image_size (Tuple): The output image size.
            extra_pixels (int): Extra pixels to add to the input image size. Defaults to 0.
        """
        super(TensorflowIdentityImagePipeline, self, ).__init__(output_image_size, extra_pixels)
        self.extra_pixels = extra_pixels
        self.output_image_size = output_image_size

    def get_image_input_size(self) -> Tuple:
        """
        Get the size of the input image considering extra pixels.

        Returns:
            Tuple: Size of the input image.
        """
        return tuple(np.array(self.output_image_size) + self.extra_pixels)

    def image_input_manipulation(self,
                                 images: tf.Tensor) -> tf.Tensor:
        """
        Apply image manipulation functions to input images.

        Args:
            images (tf.Tensor): Input images.

        Returns:
            tf.Tensor: Manipulated images.
        """
        return images

    def image_output_finalize(self,
                              images: tf.Tensor) -> tf.Tensor:
        """
        Apply image output finalization functions to images.

        Args:
            images (tf.Tensor): Input images.

        Returns:
            tf.Tensor: Finalized images.
        """
        return images


# Dictionary mapping ImagePipelineType to corresponding image pipeline classes
image_pipeline_dict: Dict[ImagePipelineType, Type[BaseImagePipeline]] = {
    ImagePipelineType.IDENTITY: TensorflowIdentityImagePipeline,
    ImagePipelineType.RANDOM_CROP_FLIP: TensorflowCropFlipImagePipeline
}

# Dictionary mapping ImageNormalizationType to corresponding normalization values
image_normalization_dict: Dict[ImageNormalizationType, List[List[float]]] = {
    ImageNormalizationType.TORCHVISION: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    ImageNormalizationType.KERAS_APPLICATIONS: [(127.5, 127.5, 127.5), (127.5, 127.5, 127.5)],
    ImageNormalizationType.NO_NORMALIZATION: [[0, 0, 0], [1, 1, 1]]
}
