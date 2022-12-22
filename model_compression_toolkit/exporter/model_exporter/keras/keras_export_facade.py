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

import keras
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.exporter.model_exporter.keras.fakely_quant_keras_exporter import \
    FakelyQuantKerasExporter


class KerasExportMode(Enum):
    FAKELY_QUANT = 0


def keras_export_model(model: keras.models.Model,
                       is_layer_exportable_fn: Callable,
                       mode: KerasExportMode = KerasExportMode.FAKELY_QUANT,
                       save_model_path: str = None) -> Dict[str, type]:
    """
    Prepare and return fully quantized model for export. Save exported model to
    a path if passed.

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
            f'Unsupported mode was used {mode.name} to export Keras model. Please see API for supported modes.')

    exporter.export()

    return exporter.get_custom_objects()
