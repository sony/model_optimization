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

from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder
from model_compression_toolkit.verify_packages import FOUND_ONNX
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

        def export(self, output_names=None) -> None:
            """
            Convert an exportable (fully-quantized) PyTorch model to a fakely-quant model
            (namely, weights that are in fake-quant format) and fake-quant layers for the activations.

            Returns:
                Fake-quant PyTorch model.
            """
            # When exporting using Fakely Quant Quantization Format list all activation quantization holders with
            # num_bits>8 and replace them with Identity, because ONNX doesn't support quantization of more than 8 bits
            # for torch.fake_quantize_per_tensor_affine.
            if not self._use_onnx_custom_quantizer_ops:
                act_holder_list = [n for n, m in self.model.named_modules()
                                   if isinstance(m, PytorchActivationQuantizationHolder) and
                                   m.activation_holder_quantizer.num_bits > 8]
                for act_holder in act_holder_list: # pragma: no cover
                    obj = self.model
                    attrs = act_holder.split(".")
                    for a in attrs[:-1]:
                        obj = getattr(obj, a)
                    if hasattr(obj, attrs[-1]):
                        delattr(obj, attrs[-1])
                        setattr(obj, attrs[-1], torch.nn.Identity())
                    else:
                        Logger.info(f"During removal of activation quantization of a quantizer (with bits > 8) in ONNX"
                                    f"FQ export, deletion of activation holder '{act_holder}' failed — could not locate"
                                    f"one or more intermediate attributes in the path.")

            for layer in self.model.children():
                self.is_layer_exportable_fn(layer)
                # Set reuse for weight quantizers if quantizer is reused
                if isinstance(layer, PytorchQuantizationWrapper):
                    for _, quantizer in layer.weights_quantizers.items():
                        if quantizer.reuse:
                            quantizer.enable_reuse_quantizer()

            # Set forward that is used during onnx export.
            # If _use_onnx_custom_quantizer_ops is set to True, the quantizer forward function will use
            # the custom implementation when exporting the operator into onnx model. If not, it removes the
            # wraps and quantizes the ops in place (for weights, for activation torch quantization function is
            # exported since it's used during forward).
            if self._use_onnx_custom_quantizer_ops:
                self._enable_onnx_custom_ops_export()
            else:
                self._substitute_fully_quantized_model(replace_wrapped=False)

            if self._use_onnx_custom_quantizer_ops:
                Logger.info(f"Exporting onnx model with MCTQ quantizers: {self.save_model_path}")
            else:
                Logger.info(f"Exporting fake-quant onnx model: {self.save_model_path}")

            model_input = to_torch_tensor(next(self.repr_dataset()))
            model_output = self.model(*model_input) if isinstance(model_input, (list, tuple)) else self.model(
                model_input)

            input_names = [f"input_{i}" for i in range(len(model_input))] if len(model_input) > 1 else ["input"]
            dynamic_axes = {name: {0: 'batch_size'} for name in input_names}
            if output_names is None:
                # Determine number of outputs and prepare output_names and dynamic_axes
                if isinstance(model_output, (list, tuple)):
                    output_names = [f"output_{i}" for i in range(len(model_output))]
                    dynamic_axes.update({name: {0: 'batch_size'} for name in output_names})
                else:
                    output_names = ['output']
                    dynamic_axes.update({'output': {0: 'batch_size'}})
            else:
                if isinstance(model_output, (list, tuple)):
                    num_of_outputs = len(model_output)
                else:
                    num_of_outputs = 1
                assert len(output_names) == num_of_outputs, (f"Mismatch between number of requested output names "
                                                             f"({output_names}) and model output count "
                                                             f"({num_of_outputs}):\n")
                dynamic_axes.update({name: {0: 'batch_size'} for name in output_names})
            if hasattr(self.model, 'metadata'):
                onnx_bytes = BytesIO()
                torch.onnx.export(self.model,
                                  tuple(model_input) if isinstance(model_input, list) else model_input,
                                  onnx_bytes,
                                  opset_version=self._onnx_opset_version,
                                  verbose=False,
                                  input_names=input_names,
                                  output_names=output_names,
                                  dynamic_axes=dynamic_axes)
                onnx_model = onnx.load_from_string(onnx_bytes.getvalue())
                onnx_model = add_onnx_metadata(onnx_model, self.model.metadata)
                onnx.save_model(onnx_model, self.save_model_path)
            else:
                torch.onnx.export(self.model,
                                  tuple(model_input) if isinstance(model_input, list) else model_input,
                                  self.save_model_path,
                                  opset_version=self._onnx_opset_version,
                                  verbose=False,
                                  input_names=input_names,
                                  output_names=output_names,
                                  dynamic_axes=dynamic_axes)

            for layer in self.model.children():
                # Set disable for reuse for weight quantizers if quantizer was reused
                if isinstance(layer, PytorchQuantizationWrapper):
                    for _, quantizer in layer.weights_quantizers.items():
                        if quantizer.reuse:
                            quantizer.disable_reuse_quantizer()

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
    def FakelyQuantONNXPyTorchExporter(*args, **kwargs):  # pragma: no cover
        Logger.critical("ONNX must be installed to use 'FakelyQuantONNXPyTorchExporter'. "
                        "The 'onnx' package is missing.")
