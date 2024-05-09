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
from io import BytesIO

import torch.nn

from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper
from model_compression_toolkit.constants import FOUND_ONNX
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.exporter.model_exporter.pytorch.base_pytorch_exporter import BasePyTorchExporter
from mct_quantizers import pytorch_quantizers


if FOUND_ONNX:
    import onnx
    from mct_quantizers.pytorch.metadata import add_onnx_metadata

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
                     onnx_opset_version: int,
                     use_onnx_custom_quantizer_ops: bool = False):
            """

            Args:
                model: Model to export.
                is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
                save_model_path: Path to save the exported model.
                repr_dataset: Representative dataset (needed for creating torch script).
                onnx_opset_version: ONNX opset version to use for exported ONNX model.
                use_onnx_custom_quantizer_ops: Whether to export quantizers custom ops in ONNX or not.
            """

            super().__init__(model,
                             is_layer_exportable_fn,
                             save_model_path,
                             repr_dataset)

            self._use_onnx_custom_quantizer_ops = use_onnx_custom_quantizer_ops
            self._onnx_opset_version = onnx_opset_version

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
            # exported since it's used during forward).
            if self._use_onnx_custom_quantizer_ops:
                self._enable_onnx_custom_ops_export()
            else:
                self._substitute_fully_quantized_model()

            if self._use_onnx_custom_quantizer_ops:
                Logger.info(f"Exporting onnx model with MCTQ quantizers: {self.save_model_path}")
            else:
                Logger.info(f"Exporting fake-quant onnx model: {self.save_model_path}")

            model_input = to_torch_tensor(next(self.repr_dataset())[0])

            if hasattr(self.model, 'metadata'):
                onnx_bytes = BytesIO()
                torch.onnx.export(self.model,
                                  model_input,
                                  onnx_bytes,
                                  opset_version=self._onnx_opset_version,
                                  verbose=False,
                                  input_names=['input'],
                                  output_names=['output'],
                                  dynamic_axes={'input': {0: 'batch_size'},
                                                'output': {0: 'batch_size'}})
                onnx_model = onnx.load_from_string(onnx_bytes.getvalue())
                onnx_model = add_onnx_metadata(onnx_model, self.model.metadata)
                onnx.save_model(onnx_model, self.save_model_path)
            else:
                torch.onnx.export(self.model,
                                  model_input,
                                  self.save_model_path,
                                  opset_version=self._onnx_opset_version,
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

            for n, m in self.model.named_modules():
                if isinstance(m, PytorchActivationQuantizationHolder):
                    assert isinstance(m.activation_holder_quantizer, pytorch_quantizers.BasePyTorchInferableQuantizer)
                    m.activation_holder_quantizer.enable_custom_impl()

                if isinstance(m, PytorchQuantizationWrapper):
                    for wq in m.weights_quantizers.values():
                        assert isinstance(wq, pytorch_quantizers.BasePyTorchInferableQuantizer)
                        wq.enable_custom_impl()

else:
    def FakelyQuantONNXPyTorchExporter(*args, **kwargs):
        Logger.critical("ONNX must be installed to use 'FakelyQuantONNXPyTorchExporter'. "
                        "The 'onnx' package is missing.")  # pragma: no cover
