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

import torch.nn

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_onnx_pytorch_exporter import \
    FakelyQuantONNXPyTorchExporter
from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_torchscript_pytorch_exporter import \
    FakelyQuantTorchScriptPyTorchExporter


class PyTorchExportMode(Enum):
    FAKELY_QUANT_TORCHSCRIPT = 0
    FAKELY_QUANT_ONNX = 1


def pytorch_export_model(model: torch.nn.Module,
                         is_layer_exportable_fn: Callable,
                         mode: PyTorchExportMode = PyTorchExportMode.FAKELY_QUANT_TORCHSCRIPT,
                         save_model_path: str = None,
                         repr_dataset: Callable = None) -> None:
    """
    Prepare and return fully quantized model for export. Save exported model to
    a path if passed.

    Args:
        model: Model to export.
        is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
        mode: Mode to export the model according to.
        save_model_path: Path to save the model.
        repr_dataset: Representative dataset for tracing the pytorch model (mandatory for exporting it).

    """

    if mode == PyTorchExportMode.FAKELY_QUANT_TORCHSCRIPT:
        exporter = FakelyQuantTorchScriptPyTorchExporter(model,
                                                         is_layer_exportable_fn,
                                                         save_model_path,
                                                         repr_dataset)

    elif mode == PyTorchExportMode.FAKELY_QUANT_ONNX:
        exporter = FakelyQuantONNXPyTorchExporter(model,
                                                  is_layer_exportable_fn,
                                                  save_model_path,
                                                  repr_dataset)

    else:
        Logger.critical(
            f'Unsupported mode was used {mode.name} to export PyTorch model. Please see API for supported modes.')

    exporter.export()
