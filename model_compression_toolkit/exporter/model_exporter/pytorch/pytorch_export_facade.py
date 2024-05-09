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

from model_compression_toolkit.constants import FOUND_TORCH
from model_compression_toolkit.exporter.model_exporter.fw_agonstic.quantization_format import QuantizationFormat
from model_compression_toolkit.exporter.model_exporter.pytorch.export_serialization_format import \
    PytorchExportSerializationFormat
from model_compression_toolkit.logger import Logger


DEFAULT_ONNX_OPSET_VERSION = 15


if FOUND_TORCH:
    import torch.nn
    from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_onnx_pytorch_exporter import FakelyQuantONNXPyTorchExporter
    from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_torchscript_pytorch_exporter import FakelyQuantTorchScriptPyTorchExporter
    from model_compression_toolkit.exporter.model_wrapper.pytorch.validate_layer import is_pytorch_layer_exportable

    supported_serialization_quantization_export_dict = {
        PytorchExportSerializationFormat.TORCHSCRIPT: [QuantizationFormat.FAKELY_QUANT],
        PytorchExportSerializationFormat.ONNX: [QuantizationFormat.FAKELY_QUANT, QuantizationFormat.MCTQ]
    }

    def pytorch_export_model(model: torch.nn.Module,
                             save_model_path: str,
                             repr_dataset: Callable,
                             is_layer_exportable_fn: Callable = is_pytorch_layer_exportable,
                             serialization_format: PytorchExportSerializationFormat = PytorchExportSerializationFormat.ONNX,
                             quantization_format: QuantizationFormat = QuantizationFormat.MCTQ,
                             onnx_opset_version=DEFAULT_ONNX_OPSET_VERSION) -> None:
        """
        Export a PyTorch quantized model to a torchscript or onnx model.
        The model will be saved to the path in save_model_path.
        Currently, pytorch_export_model supports only QuantizationFormat.FAKELY_QUANT (where weights
        and activations are float fakely-quantized values) and PytorchExportSerializationFormat.TORCHSCRIPT
        (where the model will be saved to TorchScript model) or PytorchExportSerializationFormat.ONNX
        (where the model will be saved to ONNX model).

        Args:
            model: Model to export.
            save_model_path: Path to save the model.
            repr_dataset: Representative dataset for tracing the pytorch model (mandatory for exporting it).
            is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
            serialization_format: Format to export the model according to (by default
            PytorchExportSerializationFormat.ONNX).
            quantization_format: Format of how quantizers are exported (fakely-quant, int8, MCTQ quantizers).
            onnx_opset_version: ONNX opset version to use for exported ONNX model.

        """

        if serialization_format == PytorchExportSerializationFormat.TORCHSCRIPT:
            if quantization_format in supported_serialization_quantization_export_dict[serialization_format]:
                exporter = FakelyQuantTorchScriptPyTorchExporter(model,
                                                                 is_layer_exportable_fn,
                                                                 save_model_path,
                                                                 repr_dataset)
            else:
                Logger.critical(
                    f'Unsupported quantization {quantization_format} for '
                    f'serialization {serialization_format} was used to export Pytorch model. Please see API for '
                    f'supported formats.')  # pragma: no cover

        elif serialization_format == PytorchExportSerializationFormat.ONNX:
            if quantization_format == QuantizationFormat.FAKELY_QUANT:
                exporter = FakelyQuantONNXPyTorchExporter(model,
                                                          is_layer_exportable_fn,
                                                          save_model_path,
                                                          repr_dataset,
                                                          onnx_opset_version=onnx_opset_version)
            elif quantization_format == QuantizationFormat.MCTQ:
                exporter = FakelyQuantONNXPyTorchExporter(model,
                                                          is_layer_exportable_fn,
                                                          save_model_path,
                                                          repr_dataset,
                                                          use_onnx_custom_quantizer_ops=True,
                                                          onnx_opset_version=onnx_opset_version)
            else:
                Logger.critical(
                    f'Unsupported quantization {quantization_format} for '
                    f'serialization {serialization_format} was used to export Pytorch model. Please see API for '
                    f'supported formats.')  # pragma: no cover

        else:
            Logger.critical(
                f'Unsupported serialization {serialization_format} was used to export Pytorch model.'
                f' Please see API for supported formats.')  # pragma: no cover

        exporter.export()

else:
    def pytorch_export_model(*args, **kwargs):
        Logger.critical("PyTorch must be installed to use 'pytorch_export_model'. "
                        "The 'torch' package is missing.")  # pragma: no cover
