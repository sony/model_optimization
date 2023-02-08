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
from enum import Enum
from typing import Callable, Dict

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import FOUND_TF


class KerasExportMode(Enum):
    FAKELY_QUANT = 0


if FOUND_TF:
    import keras
    from model_compression_toolkit.exporter.model_wrapper.keras.validate_layer import is_keras_layer_exportable
    from model_compression_toolkit.exporter.model_exporter.keras.fakely_quant_keras_exporter import FakelyQuantKerasExporter

    def keras_export_model(model: keras.models.Model,
                           save_model_path: str,
                           is_layer_exportable_fn: Callable = is_keras_layer_exportable,
                           mode: KerasExportMode = KerasExportMode.FAKELY_QUANT) -> Dict[str, type]:
        """
        Export a Keras quantized model to h5 model.
        The model will be saved to the path in save_model_path.
        Mode can be used for different exported files. Currently, keras_export_model
        supports KerasExportMode.FAKELY_QUANT (where weights and activations are
        float fakely-quantized values).

        Args:
            model: Model to export.
            is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
            mode: Mode to export the model according to.
            save_model_path: Path to save the model.

        Returns:
            Custom objects dictionary needed to load the model.

        """

        if mode == KerasExportMode.FAKELY_QUANT:
            exporter = FakelyQuantKerasExporter(model,
                                                is_layer_exportable_fn,
                                                save_model_path)

        else:
            Logger.critical(
                f'Unsupported mode was used {mode.name} to '
                f'export Keras model. Please see API for supported modes.')  # pragma: no cover

        exporter.export()

        return exporter.get_custom_objects()
else:
    def keras_export_model(*args, **kwargs):
        Logger.error('Installing tensorflow and tensorflow_model_optimization is mandatory '
                     'when using keras_export_model. '
                     'Could not find some or all of TensorFlow packages.')  # pragma: no cover
