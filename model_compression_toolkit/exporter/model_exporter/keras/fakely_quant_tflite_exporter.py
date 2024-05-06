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
import os
from pathlib import Path
import tempfile
from typing import Callable

import tensorflow as tf
import keras.models

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.exporter.model_exporter.keras.fakely_quant_keras_exporter import FakelyQuantKerasExporter
from model_compression_toolkit.trainable_infrastructure.keras.load_model import keras_load_quantized_model
from model_compression_toolkit.exporter.model_exporter.keras.base_keras_exporter import DEFAULT_KERAS_EXPORT_EXTENTION

class FakelyQuantTFLiteExporter(FakelyQuantKerasExporter):
    """
    Exporter for fakely-quant TFLite models.
    The exporter expects to receive an exportable model (where each layer's full quantization parameters
    can be retrieved), and convert it into a fakely-quant model (namely, weights that are in fake-quant
    format) and fake-quant layers for the activations.
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

        self.exported_model = None

    def export(self):
        """
        Convert an exportable (fully-quantized) Keras model to a fakely-quant TFLite model
        (namely, weights that are in fake-quant format) and fake-quant layers for the activations.

        """
        # Delete metadata layer if exists
        if hasattr(self.model, 'metadata_layer'):
            Logger.info('Metadata is not exported to TFLite models.')
            delattr(self.model, 'metadata_layer')

        # Use Keras exporter to quantize model's weights before converting it to TFLite.
        # Since exporter saves the model, we use a tmp path for saving, and then we delete it.
        handle, tmp_file = tempfile.mkstemp(DEFAULT_KERAS_EXPORT_EXTENTION)
        # Close handle right away, the file is going to be reopenned by Keras exporter
        os.close(handle)
        try:
            custom_objects = FakelyQuantKerasExporter(self.model,
                                                      self.is_layer_exportable_fn,
                                                      tmp_file,
                                                      verbose=False).export()

            model = keras_load_quantized_model(tmp_file)
        # Ensures artifact is removed even in case of error
        finally:
            Path(tmp_file).unlink(missing_ok=True)

        self.exported_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
        Logger.info(f'Exporting FQ tflite model to: {self.save_model_path}')
        with open(self.save_model_path, 'wb') as f:
            f.write(self.exported_model)
