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
from typing import Callable

import keras

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.exporter.model_exporter.tflite.fakely_quant_tflite_exporter import \
    FakelyQuantTFLiteExporter


class TFLiteExportMode(Enum):
    FAKELY_QUANT = 0


def tflite_export_model(model: keras.models.Model,
                        is_layer_exportable_fn: Callable,
                        mode: TFLiteExportMode = TFLiteExportMode.FAKELY_QUANT,
                        save_model_path: str = None):
    """
    Prepare and return fully quantized model for export. Save exported model to
    a path if passed.

    Args:
        model: Model to export.
        is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
        mode: Mode to export the model according to.
        save_model_path: Path to save the model.

    """

    if mode == TFLiteExportMode.FAKELY_QUANT:
        exporter = FakelyQuantTFLiteExporter(model,
                                             is_layer_exportable_fn,
                                             save_model_path)

    else:
        Logger.critical(
            f'Unsupported mode was used {mode.name} to export TFLite model.'
            f' Please see API for supported modes.')

    exporter.export()
