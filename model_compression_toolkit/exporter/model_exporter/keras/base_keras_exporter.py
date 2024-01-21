# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Callable

from model_compression_toolkit.exporter.model_exporter.fw_agonstic.exporter import Exporter
import keras

import tensorflow as tf
from packaging import version

DEFAULT_KERAS_EXPORT_EXTENTION = '.keras'


class BaseKerasExporter(Exporter):
    """
    Base Keras exporter class.
    """

    def __init__(self,
                 model: keras.models.Model,
                 is_layer_exportable_fn: Callable,
                 save_model_path: str):
        """
        Args:
            model: Model to export.
            is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
            save_model_path: Path to save the exported model.

        """
        super().__init__(model,
                         is_layer_exportable_fn,
                         save_model_path)

    @staticmethod
    def get_custom_objects():
        return {}
