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

import torch.nn

from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.exporter.model_exporter.pytorch.base_pytorch_exporter import BasePyTorchExporter
from packaging import version
from mct_quantizers import pytorch_quantizers

# ONNX opset version 16 is supported from PyTorch 1.12
if version.parse(torch.__version__) < version.parse("1.12"):
    OPSET_VERSION = 15
else:
    OPSET_VERSION = 16


class FakelyQuantONNXPyTorchExporter(BasePyTorchExporter):
    """
    Exporter for fakely-quant PyTorch models.
    The exporter expects to receive an exportable model (where each layer's full quantization parameters
    can be retrieved), and convert it into a fakely-quant model (namely, weights that are in fake-quant
    format) and fake-quant layers for the activations.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 is_layer_exportable_fn: Callable,
                 save_model_path: str,
                 repr_dataset: Callable,
                 use_onnx_custom_quantizer_ops: bool = False):
        """

        Args:
            model: Model to export.
            is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
            save_model_path: Path to save the exported model.
            repr_dataset: Representative dataset (needed for creating torch script).
            use_onnx_custom_quantizer_ops: Whether to export quantizers custom ops in ONNX or not.
        """

        super().__init__(model,
                         is_layer_exportable_fn,
                         save_model_path,
                         repr_dataset)

        self._use_onnx_custom_quantizer_ops = use_onnx_custom_quantizer_ops


    def export(self) -> None:
        """
        Convert an exportable (fully-quantized) PyTorch model to a fakely-quant model
        (namely, weights that are in fake-quant format) and fake-quant layers for the activations.

        Returns:
            Fake-quant PyTorch model.
        """
        for layer in self.model.children():
            self.is_layer_exportable_fn(layer)

        # Set forward that is used during onnx export.
        # If _use_onnx_custom_quantizer_ops is set to True, the quantizer forward function will use
        # the custom implementation when exporting the operator into onnx model. If not, it removes the
        # wraps and quantizes the ops in place (for weights, for activation torch quantization function is
        # exported since it's used during forward.
        if self._use_onnx_custom_quantizer_ops:
            self._enable_onnx_custom_ops_export()
        else:
            self._substitute_fully_quantized_model()

        Logger.info(f"Exporting PyTorch fake quant onnx model: {self.save_model_path}")

        model_input = to_torch_tensor(next(self.repr_dataset())[0])

        torch.onnx.export(self.model,
                          model_input,
                          self.save_model_path,
                          opset_version=OPSET_VERSION,
                          verbose=False,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

    def _enable_onnx_custom_ops_export(self):
        """
        Enable the custom implementation forward in quantizers, so it is exported
        with custom quantizers.
        """

        for n, m in self.model.named_children():
            if isinstance(m, PytorchActivationQuantizationHolder):
                assert isinstance(m.activation_holder_quantizer, pytorch_quantizers.BasePyTorchInferableQuantizer)
                m.activation_holder_quantizer.enable_custom_impl()

            if isinstance(m, PytorchQuantizationWrapper):
                for wq in m.weights_quantizers.values():
                    assert isinstance(wq, pytorch_quantizers.BasePyTorchInferableQuantizer)
                    wq.enable_custom_impl()
