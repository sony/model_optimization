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
import timm_lib
from timm.data import create_dataset, create_loader, resolve_data_config

from tutorials.quick_start.common.model_lib import BaseModelLib
from tutorials.quick_start.pytorch_fw.utils import classification_eval, get_representative_dataset
from tutorials.quick_start.common.constants import MODEL_NAME, BATCH_SIZE, VALIDATION_SET_LIMIT, \
    VALIDATION_DATASET_FOLDER, IMAGENET_DATASET
from tutorials.quick_start.common.results import DatasetInfo


class ModelLib(BaseModelLib):
    """
    A class representing timm model library (https://github.com/huggingface/pytorch-image-models/tree/main).
    """

    def __init__(self, args):
        """
        Init the ModelLib with user arguments
        Args:
            args (dict): user arguments
        """
        avialable_models = timm_lib.list_models('')
        model_name = args[MODEL_NAME]
        if model_name in avialable_models:
            self.model = timm_lib.create_model(args[MODEL_NAME], pretrained=True)
            self.data_config = resolve_data_config([], model=self.model)  # include the pre-processing
            self.dataset_name = IMAGENET_DATASET
            super().__init__(args)
        else:
            raise Exception(f'Unknown timm model name {model_name}, Available models : {avialable_models}')

    def get_model(self):
        """
         Returns the model instance.
         """
        return self.model

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size):
        """
        Create a representative dataset generator
        Args:
            representative_dataset_folder: Dataset location folder, in the format: representative_dataset_folder/<class>/<images>
            n_iter: number batches to run in the generator
            batch_size: number of images in each batch

        Returns:
            A generator for the representative dataset, as the MCT expects

        """
        train_dataset = create_dataset(name='', root=representative_dataset_folder,
                                       is_training=False, batch_size=batch_size)
        dl = create_loader(
            train_dataset,
            input_size=self.data_config['input_size'],
            batch_size=batch_size,
            interpolation=self.data_config['interpolation'],
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            crop_pct=self.data_config['crop_pct'])
        return get_representative_dataset(dl, n_iter)

    def evaluate(self, model):
        """
        Evaluates the model's performance.

        Args:
            model: The model to evaluate.

        Returns:
            acc: The accuracy of the model.
            DatasetInfo: Information about the dataset used.
        """
        batch_size = int(self.args[BATCH_SIZE])
        validation_dataset_folder = self.args[VALIDATION_DATASET_FOLDER]
        val_dataset = create_dataset(name='', root=validation_dataset_folder, is_training=False,
                                           batch_size=batch_size)
        testloader = create_loader(
            val_dataset,
            input_size=self.data_config['input_size'],
            batch_size=batch_size,
            interpolation=self.data_config['interpolation'],
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            crop_pct=self.data_config['crop_pct'])

        acc, total = classification_eval(model, testloader, self.args[VALIDATION_SET_LIMIT])
        dataset_info = DatasetInfo(self.dataset_name, total)
        return acc, dataset_info



