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

import keras.models
import tensorflow as tf

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.exporter.model_exporter.keras.base_keras_exporter import BaseKerasExporter
from model_compression_toolkit.exporter.model_exporter.keras.fakely_quant_keras_exporter import FakelyQuantKerasExporter


class FakelyQuantTFLiteExporter(FakelyQuantKerasExporter):
    """
    Exporter for fakely-quant TFLite models.
    The exporter expects to receive an exportable model (where each layer's full quantization parameters
    can be retrieved), and convert it into a fakely-quant model (namely, weights that are in fake-quant
    format) and fake-quant layers for the activations.
    """

    def __init__(self,
                 model: keras.models.Model,
                 is_layer_exportable_fn: Callable):

        super().__init__(model,
                         is_layer_exportable_fn)
        self.exported_model = None

    def export(self) -> keras.models.Model:
        """
        Convert an exportable (fully-quantized) Keras model to a fakely-quant model
        (namely, weights that are in fake-quant format) and fake-quant layers for the activations.

        Returns:
            Fake-quant Keras model.
        """
        model = super(FakelyQuantTFLiteExporter, self).export()
        self.exported_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
        return self.exported_model

    def save_model(self, save_model_path: str) -> None:
        """
        Save exported model to a given path.
        Args:
            save_model_path: Path to save the model.

        Returns:
            None.
        """
        if self.exported_model is None:
            Logger.critical(f'Exporter can not save model as it is not exported')
        Logger.info(f'Exporting FQ tflite model to: {save_model_path}')
        with open(save_model_path, 'wb') as f:
            f.write(self.exported_model)
