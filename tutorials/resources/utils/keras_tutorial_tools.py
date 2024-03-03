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


import tensorflow as tf
import keras
import model_compression_toolkit as mct
import os
import torchvision
import subprocess
import tarfile
from typing import Generator



def get_validation_dataset_fraction(fraction, test_dataset_folder, batch) -> tf.data.Dataset:
    """
    Load a fraction of the validation dataset for evaluation.

    Args:
        fraction (float, optional): Fraction of the dataset to load. Defaults to 1.0 (i.e., the entire dataset).

    Returns:
        tf.data.Dataset: A fraction of the validation dataset.
    """
    assert 0 < fraction <= 1, "Fraction must be between 0 and 1."

    # Load the dataset to determine the total number of samples
    initial_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=test_dataset_folder,
        batch_size=1,  # Use batch size of 1 to count samples
        image_size=[224, 224],
        shuffle=False,
        crop_to_aspect_ratio=True,
        interpolation='bilinear')

    total_samples = initial_dataset.cardinality().numpy()
    samples_to_take = int(total_samples * fraction)

    # reload the dataset again with batch size + take number of samples
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DATASET_FOLDER,
        batch_size=batch,
        image_size=[224, 224],
        shuffle=False,
        crop_to_aspect_ratio=True,
        interpolation='bilinear')

    # Preprocess the dataset
    dataset = dataset.map(lambda x, y: (imagenet_preprocess_input(x, y)))

    # Take the calculated number of samples (adjusted for batch size)
    dataset = dataset.take(samples_to_take // batch + (1 if samples_to_take % batch else 0))

    return dataset


def get_representative_dataset(fraction, representative_dataset_folder, batch) -> Generator:
    """
    A function that loads a fraction of the dataset and returns a representative dataset generator.

    Args:
        fraction (float): The fraction of the dataset to load. Defaults to 1.0 (the entire dataset).

    Returns:
        Generator: A generator yielding batches of preprocessed images.
    """
    assert 0 < fraction <= 1, "Fraction must be between 0 and 1."

    print('Loading dataset, this may take a few minutes ...')    
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=representative_dataset_folder,
        batch_size=batch,
        image_size=[224, 224],
        shuffle=True,
        crop_to_aspect_ratio=True,
        interpolation='bilinear')

    # Preprocess the data
    dataset = dataset.map(lambda x, y: (imagenet_preprocess_input(x, y)))

    # Determine the total number of batches in the dataset
    total_batches = dataset.cardinality().numpy()
    if total_batches == tf.data.experimental.INFINITE_CARDINALITY:
        raise ValueError("Dataset size is infinite. A finite dataset is required to compute a fraction.")

    # Calculate the number of batches to use, based on the specified fraction
    batches_to_use = int(total_batches * fraction)

    def representative_dataset() -> Generator:
        """A generator function that yields batches of preprocessed images."""
        for image_batch, _ in dataset.take(batches_to_use):
            yield image_batch.numpy()

    print('images in representative dataset: '+ str(batch*batches_to_use))

    return representative_dataset


