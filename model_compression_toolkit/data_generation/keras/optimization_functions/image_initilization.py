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
from typing import Tuple, Any

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

from model_compression_toolkit.data_generation.common.constants import NUM_INPUT_CHANNELS
from model_compression_toolkit.data_generation.common.enums import DataInitType
from model_compression_toolkit.logger import Logger


# Define a function to generate a dataset of Gaussian noise images.
def generate_gaussian_noise_images(mean: float,
                                   std: float,
                                   num_samples: int,
                                   batch_size: int,
                                   image_shape: Tuple[int, int, int]) -> Dataset:
    """
    Generates a dataset of Gaussian noise images.

    Args:
        mean (float): Mean value for the Gaussian noise.
        std (float): Standard deviation for the Gaussian noise.
        num_samples (int): Total number of noise samples to generate.
        batch_size (int): Batch size for the dataset.
        image_shape (Tuple[int, int, int]): Shape of each noise image.

    Returns:
        tf.data.Dataset: A TensorFlow dataset containing Gaussian noise images.
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: noise_generator(mean, std, num_samples, image_shape),
        output_signature=tf.TensorSpec(shape=image_shape, dtype=tf.float32)
    )
    dataset = dataset.batch(batch_size)
    return dataset


# Generator function for Gaussian noise images.
def noise_generator(mean: float,
                    std: float,
                    total_images: int,
                    image_shape: Tuple[int, int, int]):
    """
    Generator function for generating Gaussian noise images.

    Args:
        mean (float): Mean value for the Gaussian noise.
        std (float): Standard deviation for the Gaussian noise.
        total_images (int): Total number of noise images to generate.
        image_shape (Tuple[int, int, int]): Shape of each noise image.

    Yields:
        np.ndarray: A Gaussian noise image.
    """
    for _ in range(total_images):
        noise = np.random.normal(mean, std, size=image_shape).astype(np.float32)
        yield noise


def get_random_gaussian_data(
        batch_size=1,
        n_images=10000,
        size: Tuple = (224, 224),
        mean_factor=0,
        std_factor=1) -> Tuple[int, Any]:
    """
    Generates random Gaussian data.

    Args:
        batch_size (int): Batch size for the dataset.
        n_images (int): Total number of noise images to generate.
        size (Tuple[int, int]): Size of each image as (height, width).
        mean_factor (float): Factor to scale the mean of the Gaussian noise.
        std_factor (float): Factor to scale the standard deviation of the Gaussian noise.

    Returns:
        Tuple[int, Any]: A tuple containing the number of batches and a data loader iterator.
    """
    Logger.info(f'Start generating random Gaussian data')
    image_shape = size + (NUM_INPUT_CHANNELS,)
    dataset = generate_gaussian_noise_images(num_samples=n_images, image_shape=image_shape,
                                             mean=mean_factor, std=std_factor, batch_size=batch_size)
    data_loader = iter(dataset)
    return data_loader


# Dictionary of image initialization functions
image_initialization_function_dict = {
    DataInitType.Gaussian: partial(get_random_gaussian_data)
}
