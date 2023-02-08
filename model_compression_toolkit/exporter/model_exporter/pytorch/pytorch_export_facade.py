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

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import FOUND_TORCH


class PyTorchExportMode(Enum):
    FAKELY_QUANT_TORCHSCRIPT = 0
    FAKELY_QUANT_ONNX = 1


if FOUND_TORCH:
    import torch.nn
    from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_onnx_pytorch_exporter import FakelyQuantONNXPyTorchExporter
    from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_torchscript_pytorch_exporter import FakelyQuantTorchScriptPyTorchExporter
    from model_compression_toolkit.exporter.model_wrapper.pytorch.validate_layer import is_pytorch_layer_exportable

    def pytorch_export_model(model: torch.nn.Module,
                             save_model_path: str,
                             repr_dataset: Callable,
                             is_layer_exportable_fn: Callable = is_pytorch_layer_exportable,
                             mode: PyTorchExportMode = PyTorchExportMode.FAKELY_QUANT_TORCHSCRIPT) -> None:
        """
        Export a PyTorch quantized model to a torchscript or onnx model.
        The model will be saved to the path in save_model_path.
        Mode can be used for different exported files. Currently, pytorch_export_model
        supports PyTorchExportMode.FAKELY_QUANT_TORCHSCRIPT (where the exported model
        is in a TorchScript format and its weights and activations are float fakely-quantized values),
        and PyTorchExportMode.FakelyQuantONNX (where the exported model
        is in an ONNX format and its weights and activations are float fakely-quantized values)

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
                f'Unsupported mode was used {mode.name} to export PyTorch model. '
                f'Please see API for supported modes.')  # pragma: no cover

        exporter.export()

else:
    def pytorch_export_model(*args, **kwargs):
        Logger.error('Installing torch is mandatory '
                     'when using pytorch_export_model. '
                     'Could not find PyTorch packages.')  # pragma: no cover
