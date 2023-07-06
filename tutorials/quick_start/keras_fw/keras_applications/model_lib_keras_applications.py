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

from typing import Dict
import tensorflow as tf
import keras

from tutorials.quick_start.common.model_lib import BaseModelLib
from tutorials.quick_start.keras_fw.utils import classification_eval, get_representative_dataset, separate_preprocess_model
from tutorials.quick_start.common.constants import MODEL_NAME, BATCH_SIZE, VALIDATION_SET_LIMIT, VALIDATION_DATASET_FOLDER, IMAGENET_DATASET

from tutorials.quick_start.common.results import DatasetInfo


class ModelLib(BaseModelLib):
    """
    A class for quantizing pre-trained models from the tensorflow.keras.applications library
    """
    def __init__(self, args: Dict):
        """
        Init the ModelLib with user arguments
        Args:
            args (dict): user arguments
        """
        model_fn, self.model_package = self.get_keras_apps_model(args[MODEL_NAME])
        # create model
        self.model = model_fn()
        self.preprocess = self.generate_preprocess()
        self.dataset_name = IMAGENET_DATASET
        # Extract image input size and verify it's not dynamic (equals None)
        self.input_size = self.model.layers[0].input_shape[0][1:3]
        if None in self.input_size:
            raise Exception(f'model {args[MODEL_NAME]} has a dynamic input size, which is not supported by the MCT')
        super().__init__(args)

    def get_keras_apps_model(self, model_name: str):
        """
        Extracts the model package and model name from the input string: <package>.<model>
        The model package class contains the 'preprocess_input' method for preprocessing input images
        One exception is the MobileNetV3 models that are located in the parent model (tf.keras.applications)

        Args:
            model_name (str): A string containing the model to quantize in the following format: <package>.<model> (e.g. mobilenet_v2.MobileNetV2)

        Returns:
            model package class
            model class

        """
        if hasattr(tf.keras.applications, model_name):
            model_class = getattr(tf.keras.applications, model_name)
            model_package = eval(model_class.__module__)
        else:
            raise Exception(f'Unknown Keras Applications model class {model_name}')

        return model_class, model_package

    @staticmethod
    def get_keras_apps_weights(model_name):
        return None

    def generate_preprocess(self, truncate_preprocess: bool = False):
        """
        Generates the preprocess function for evaluation and quantization (representative dataset)
        Some models in this library contain the normalization of the preprocess in the beginning of the model. They can be removed.
        Args:
            truncate_preprocess (bool): Removes Normalization & Rescaling layers from the beginning of the model

        Returns:

        """
        pp_model = None
        if truncate_preprocess:
            self.model, pp_model = separate_preprocess_model(self.model)

        def _preprocess(x, l):
            x = self.model_package.preprocess_input(x)
            if pp_model is not None:
                x = pp_model(x)
            return x, l

        return _preprocess

    def get_model(self):
        return self.model

    def get_representative_dataset(self, representative_dataset_folder: str, n_iter: int, batch_size: int):
        """
        Create a representative dataset generator
        Args:
            representative_dataset_folder: Dataset location folder, in the format: representative_dataset_folder/<class>/<images>
            n_iter: number batches to run in the generator
            batch_size: number of images in each batch

        Returns:
            A generator for the representative dataset, as the MCT expects

        """
        dl = tf.keras.utils.image_dataset_from_directory(representative_dataset_folder,
                                                         batch_size=batch_size,
                                                         image_size=self.input_size,
                                                         crop_to_aspect_ratio=True).map(self.preprocess)

        return get_representative_dataset(dl, n_iter)

    def evaluate(self, model):
        """
        Evaluate a model
        Args:
            model: A keras model to evaluate

        Returns:
            model accuracy
            dataset info (dataset name, number of images in dataset)

        """
        batch_size = int(self.args[BATCH_SIZE])
        validation_dataset_folder = self.args[VALIDATION_DATASET_FOLDER]
        testloader = tf.keras.utils.image_dataset_from_directory(validation_dataset_folder,
                                                                 batch_size=batch_size,
                                                                 image_size=self.input_size,
                                                                 shuffle=False,
                                                                 crop_to_aspect_ratio=True)

        testloader = testloader.map(self.preprocess)

        acc, total = classification_eval(model, testloader, self.args[VALIDATION_SET_LIMIT])
        dataset_info = DatasetInfo(self.dataset_name, total)
        return acc, dataset_info
