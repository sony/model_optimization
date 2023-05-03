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

from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.exporter.model_exporter.fw_agonstic.export_serialization_format import \
    ExportSerializationFormat
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.target_platform.quantization_format import \
    QuantizationFormat

if FOUND_TF:
    import keras
    from model_compression_toolkit.exporter.model_exporter.tflite.fakely_quant_tflite_exporter import \
        FakelyQuantTFLiteExporter
    from model_compression_toolkit.exporter.model_exporter.tflite.int8_tflite_exporter import INT8TFLiteExporter
    from model_compression_toolkit.exporter.model_wrapper.keras.validate_layer import is_keras_layer_exportable


    def tflite_export_model(model: keras.models.Model,
                            save_model_path: str,
                            target_platform_capabilities: TargetPlatformCapabilities,
                            is_layer_exportable_fn: Callable = is_keras_layer_exportable,
                            serialization_format: ExportSerializationFormat = ExportSerializationFormat.TFLITE
                            ):
        """
        Export a Keras quantized model to a tflite model.
        The model will be saved to the path in save_model_path.
        Currently, tflite_export_model supports only ExportSerializationFormat.TFLITE (where the model will be saved to
        tflite model) and QuantizationFormat.FAKELY_QUANT (where weights and activations are float fakely-quantized
        values) or QuantizationFormat.INT8 (where weights and activations are represented using 8bits integers).

        Args:
            model: Model to export.
            save_model_path: Path to save the model.
            target_platform_capabilities: TargetPlatformCapabilities object that describes the desired inference target platform
            (includes quantization format).
            is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
            serialization_format: Format to export the model according to.

        """

        if target_platform_capabilities.tp_model.quantization_format == QuantizationFormat.FAKELY_QUANT and \
                serialization_format == ExportSerializationFormat.TFLITE:
            exporter = FakelyQuantTFLiteExporter(model,
                                                 is_layer_exportable_fn,
                                                 save_model_path)
        elif target_platform_capabilities.tp_model.quantization_format == QuantizationFormat.INT8 and \
                serialization_format == ExportSerializationFormat.TFLITE:
            exporter = INT8TFLiteExporter(model,
                                          is_layer_exportable_fn,
                                          save_model_path)

        else:
            Logger.critical(
                f'Unsupported quantization {target_platform_capabilities.tp_model.quantization_format} or serialization {serialization_format} '
                f'was used to export Keras model. Please see API for supported formats.')  # pragma: no cover

        exporter.export()

else:
    def tflite_export_model(*args, **kwargs):
        Logger.error('Installing tensorflow and tensorflow_model_optimization is mandatory '
                     'when using tflite_export_model. '
                     'Could not find some or all of TensorFlow packages.')  # pragma: no cover
