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
import numpy as np
from typing import Tuple

import tensorflow as tf

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
                height_crop: int,
                width_crop: int) -> tf.Tensor:
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

    # Calculate the cropping offsets
    offset_height = tf.maximum((height - height_crop) // 2, 0)
    offset_width = tf.maximum((width - width_crop) // 2, 0)

    # Crop the image
    cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, height_crop, width_crop)

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


def clip_images(images: tf.Tensor, valid_grid: tf.Tensor, reflection: bool = False) -> tf.Tensor:
    """
    Clip the images based on a valid grid.

    Args:
        images (tf.Tensor): The images to be clipped.
        valid_grid (tf.Tensor): The valid grid for clipping.
        reflection (bool): Whether to apply reflection during clipping. Defaults to False.

    Returns:
        tf.Tensor: The clipped images.
    """
    clipped_images = tf.TensorArray(tf.float32, size=images.shape[1])

    for i in range(valid_grid.shape[0]):
        channel = images[:, i, :, :]
        min_val = tf.reduce_min(valid_grid[i, :])
        max_val = tf.reduce_max(valid_grid[i, :])
        clamp = tf.clip_by_value(channel, min_val, max_val)
        if reflection:
            channel = 2 * clamp - channel
        else:
            channel = clamp
        clipped_images = clipped_images.write(i, channel)

    clipped_images = clipped_images.stack()
    return tf.transpose(clipped_images, perm=[1, 0, 2, 3])


def create_valid_grid(means, stds) -> tf.Tensor:
    """
    Create a valid grid for image normalization.

    Returns:
        tf.Tensor: The valid grid for image normalization.
    """
    # Create a pixel grid in the range 0-255, repeat for 3 color channels, and reshape
    pixel_grid = np.arange(256).repeat(3).reshape(-1, 3)

    # Transpose and add batch and channel dimensions
    pixel_grid = tf.constant(pixel_grid, dtype=tf.float32)
    pixel_grid = tf.transpose(pixel_grid, perm=[1, 0])

    # Normalize the pixel grid using the specified mean and std
    mean = tf.constant(np.array(means), dtype=tf.float32)
    std = tf.constant(np.array(stds), dtype=tf.float32)
    valid_grid = (pixel_grid - mean[: , tf.newaxis]) / std[: , tf.newaxis]

    return valid_grid

class Smoothing(tf.keras.layers.Layer):
    """
    A TensorFlow layer for applying Gaussian smoothing to an image.
    """

    def __init__(self, size: int = 3, sigma: float = 1.25):
        """
        Initialize the Smoothing layer.

        Args:
            size (int): The size of the Gaussian kernel.
            sigma (float): The standard deviation of the Gaussian kernel.
        """
        super(Smoothing, self).__init__()
        self.size = size
        self.sigma = sigma
        self.kernel = self.gaussian_kernel(size, sigma)

    def build(self, input_shape):
        """
        Build the smoothing layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        kernel = tf.reshape(self.kernel, [self.size, self.size, 1, 1])
        self.kernel = tf.tile(kernel, [1, 1, input_shape[-1], 1])

    def call(self, inputs):
        """
        Apply Gaussian smoothing to the input image.

        Args:
            inputs (tf.Tensor): The input image tensor.

        Returns:
            tf.Tensor: The smoothed image tensor.
        """
        return tf.nn.depthwise_conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')

    def gaussian_kernel(self, size: int, sigma: float) -> tf.Tensor:
        """
        Create a Gaussian kernel.

        Args:
            size (int): The size of the Gaussian kernel.
            sigma (float): The standard deviation of the Gaussian kernel.

        Returns:
            tf.Tensor: The Gaussian kernel tensor.
        """
        axis = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        x, y = tf.meshgrid(axis, axis)
        kernel = tf.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        return kernel
