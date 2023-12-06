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
from typing import Callable

import keras.models
layers = keras.layers

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.exporter.model_exporter.keras.base_keras_exporter import BaseKerasExporter

class MCTQKerasExporter(BaseKerasExporter):

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

    def export(self):
        """
        Export an exportable (fully-quantized) Keras model to a Keras model with
        MCTQ quantizers. This is done by using keras saving model function.
        """
        if self.model is None:
            Logger.critical(f'Exporter can not save None model')  # pragma: no cover

        Logger.info(f'Exporting Keras model with MCTQ custom quantizers to: {self.save_model_path}')
        keras.models.save_model(self.model, self.save_model_path)


