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
from typing import Callable, Generator

class DatasetPreprocessor:
    """
    A class for preprocessing datasets based on the model version specified.
    This class allows for the selection of preprocessing functions tailored to
    different model architectures, ensuring that input data is correctly
    normalized and prepared for model training or inference.

    Attributes:
        model_version (str): The version of the model for which the dataset is being preprocessed.
        preprocess_input (Callable): The preprocessing function selected based on the model version.
    """
    def __init__(self, model_version: str):
        """
        Initializes the DatasetPreprocessor with a specific model version.

        Args:
            model_version (str): The version of the model for which the dataset is being preprocessed.
        """
        self.model_version = model_version
        self.preprocess_input = self.get_preprocess_function()

    def get_preprocess_function(self) -> Callable:
        """
        Selects and returns the appropriate preprocessing function based on the model version.

        Returns:
            Callable: A function that can be used to preprocess input data for the specified model version.

        Raises:
            ValueError: If the model version is unsupported.
        """
        if self.model_version == 'MobileNet':
            return tf.keras.applications.mobilenet.preprocess_input
        elif self.model_version == 'MobileNetV2':
            return tf.keras.applications.mobilenet_v2.preprocess_input
        elif self.model_version == 'MobileNetV3Small':
            return tf.keras.applications.mobilenet_v3.preprocess_input
        elif self.model_version == 'EfficientNetB1':
            return tf.keras.applications.efficientnet.preprocess_input
        elif self.model_version == 'Xception':
            return tf.keras.applications.xception.preprocess_input
        elif self.model_version == 'DenseNet121':
            return tf.keras.applications.densenet.preprocess_input
        elif self.model_version == 'NASNetMobile':
            return tf.keras.applications.nasnet.preprocess_input
        else:
            raise ValueError(f"Unsupported model version: {self.model_version}")
    def preprocess_dataset(self, images, labels):
        return self.preprocess_input(images), labels



    def get_validation_dataset_fraction(self, fraction, test_dataset_folder, batch) -> tf.data.Dataset:
        """
        Load a fraction of the validation dataset for evaluation.

        Args:
            fraction (float, optional): Fraction of the dataset to load. Defaults to 1.0 (i.e., the entire dataset).
            test_dataset_folder (str): location of dataset
            batch (int): batch size when loading dataset.

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
            directory=test_dataset_folder,
            batch_size=batch,
            image_size=[224, 224],
            shuffle=False,
            crop_to_aspect_ratio=True,
            interpolation='bilinear')

        # Preprocess the dataset
        dataset = dataset.map(self.preprocess_dataset)
        # Take the calculated number of samples (adjusted for batch size)
        dataset = dataset.take(samples_to_take // batch + (1 if samples_to_take % batch else 0))

        return dataset


    def get_representative_dataset(self, fraction, representative_dataset_folder, batch) -> Generator:
        """
        A function that loads a fraction of the dataset and returns a representative dataset generator.

        Args:
            fraction (float): The fraction of the dataset to load. Defaults to 1.0 (the entire dataset).
            test_dataset_folder (str): location of dataset
            batch (int): batch size when loading dataset.

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
        dataset = dataset.map(self.preprocess_dataset)

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


