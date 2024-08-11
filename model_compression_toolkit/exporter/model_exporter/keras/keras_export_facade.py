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
from typing import Callable, Dict

from model_compression_toolkit.verify_packages import FOUND_TF
from model_compression_toolkit.exporter.model_exporter.fw_agonstic.quantization_format import QuantizationFormat
from model_compression_toolkit.logger import Logger


if FOUND_TF:
    import keras
    from model_compression_toolkit.exporter.model_wrapper.keras.validate_layer import is_keras_layer_exportable
    from model_compression_toolkit.exporter.model_exporter.keras.fakely_quant_keras_exporter import \
        FakelyQuantKerasExporter
    from model_compression_toolkit.exporter.model_exporter.keras.fakely_quant_tflite_exporter import \
        FakelyQuantTFLiteExporter
    from model_compression_toolkit.exporter.model_exporter.keras.int8_tflite_exporter import INT8TFLiteExporter
    from model_compression_toolkit.exporter.model_exporter.keras.export_serialization_format import \
        KerasExportSerializationFormat
    from model_compression_toolkit.exporter.model_exporter.keras.mctq_keras_exporter import MCTQKerasExporter

    supported_serialization_quantization_export_dict = {
        KerasExportSerializationFormat.KERAS: [QuantizationFormat.FAKELY_QUANT, QuantizationFormat.MCTQ],
        KerasExportSerializationFormat.TFLITE: [QuantizationFormat.FAKELY_QUANT, QuantizationFormat.INT8]
    }

    def keras_export_model(model: keras.models.Model,
                           save_model_path: str,
                           is_layer_exportable_fn: Callable = is_keras_layer_exportable,
                           serialization_format: KerasExportSerializationFormat = KerasExportSerializationFormat.KERAS,
                           quantization_format : QuantizationFormat = QuantizationFormat.MCTQ) -> Dict[str, type]:
        """
        Export a Keras quantized model to a .keras or .tflite format model (according to serialization_format).
        The model will be saved to the path in save_model_path.
        Models that are exported to .keras format can use quantization_format of QuantizationFormat.MCTQ or QuantizationFormat.FAKELY_QUANT.
        Models that are exported to .tflite format can use quantization_format of QuantizationFormat.INT8 or QuantizationFormat.FAKELY_QUANT.

        Args:
            model: Model to export.
            save_model_path: Path to save the model.
            is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
            serialization_format: Format to export the model according to (KerasExportSerializationFormat.KERAS, by default).
            quantization_format: Format of how quantizers are exported (MCTQ quantizers, by default).

        Returns:
            Custom objects dictionary needed to load the model.

        """

        if serialization_format == KerasExportSerializationFormat.KERAS:
            if quantization_format == QuantizationFormat.FAKELY_QUANT:
                exporter = FakelyQuantKerasExporter(model,
                                                    is_layer_exportable_fn,
                                                    save_model_path)
            elif quantization_format == QuantizationFormat.MCTQ:
                exporter = MCTQKerasExporter(model,
                                             is_layer_exportable_fn,
                                             save_model_path)

            else:
                Logger.critical(
                    f'Unsupported quantization {quantization_format} for '
                    f'serialization {serialization_format} was used to export Keras model. Please see API for '
                    f'supported formats.')  # pragma: no cover

        elif serialization_format == KerasExportSerializationFormat.TFLITE:
            if quantization_format == QuantizationFormat.FAKELY_QUANT:
                exporter = FakelyQuantTFLiteExporter(model,
                                                     is_layer_exportable_fn,
                                                     save_model_path)

            elif quantization_format == QuantizationFormat.INT8:
                exporter = INT8TFLiteExporter(model,
                                              is_layer_exportable_fn,
                                              save_model_path)
            else:
                Logger.critical(
                    f'Unsupported quantization {quantization_format} for '
                    f'serialization {serialization_format} was used to export Keras model. Please see API for '
                    f'supported formats.')  # pragma: no cover

        else:
            Logger.critical(
                f'Unsupported serialization {serialization_format} was used to export Keras model. Please see API '
                f'for supported formats.')  # pragma: no cover

        exporter.export()

        return exporter.get_custom_objects()
else:
    def keras_export_model(*args, **kwargs):
        Logger.critical("Tensorflow must be installed with a version of 2.15 or lower to use keras_export_model."
                        "The 'tensorflow' package is missing or is installed "
                        "with a version higher than 2.15.")  # pragma: no cover
