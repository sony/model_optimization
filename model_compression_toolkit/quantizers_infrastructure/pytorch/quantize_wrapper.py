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
# ==============================================================================f
from typing import List, Union, Any, Dict
from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.quantizers_infrastructure.common.node_quantization_dispatcher import \
    NodeQuantizationDispatcher
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import BaseInferableQuantizer
from model_compression_toolkit import quantizers_infrastructure as qi
import inspect


if FOUND_TORCH:
    import torch
    import torch.nn as nn

    DISPATCHER = "dispatcher"
    LAYER = "layer"
    STEPS = "optimizer_step"
    TRAINING = "training"


    class PytorchQuantizationWrapper(nn.Module):
        def __init__(self,
                     module: nn.Module,
                     dispatcher: NodeQuantizationDispatcher):
            """
            Pytorch Quantization Wrapper takes a pytorch module and dispatcher and infer a quantized module.

            Args:
                module: A pytorch module.
                dispatcher: A node quantization dispatcher.
            """
            super().__init__()
            self._dispatcher = dispatcher
            if isinstance(module, nn.Module):
                self.add_module(LAYER, module)
            else:
                # Functional layers
                setattr(self, LAYER, module)

            # Init weight quantizers from dispatcher
            self._set_weight_quantizers(True)

            # Init activations quantizers from dispatcher
            self._set_activation_quantizers()

        def convert_to_inferable_quantizers(self):
            """
            Convert the wrapper quantizers with inferable quantizers

            """
            # Activation quantizers
            if self._dispatcher.is_activation_quantization:
                inferable_activation_quantizers = []
                for quantizer in self._dispatcher.activation_quantizers:
                    if isinstance(quantizer, qi.BasePytorchTrainableQuantizer):
                        inferable_activation_quantizers.append(quantizer.convert2inferable())
                    else:
                        Logger.error('Can only convert trainable quantizers based on BasePytorchTrainableQuantizer')
                self._dispatcher.set_activation_quantizers(inferable_activation_quantizers)
                self._set_activation_quantizers()

            # Weight quantizers
            if self._dispatcher.is_weights_quantization:
                inferable_weight_quantizers = {}
                for name, quantizer in self._dispatcher.weight_quantizers.items():
                    if isinstance(quantizer, qi.BasePytorchTrainableQuantizer):
                        inferable_weight_quantizers.update({name: quantizer.convert2inferable()})
                    else:
                        Logger.error('Can only convert trainable quantizers based on BasePytorchTrainableQuantizer')
                self._dispatcher.set_weight_quantizers(inferable_weight_quantizers)
                self._set_weight_quantizers(False)

        def _set_weight_quantizers(self, is_training: bool):
            """
            Initialize learnable weights as parameters in the wrapper, and their quantizers

            Args:
                is_training: Whether working with InferableQuantizers or not. If so, do not register weight as parameter.

            """
            self._weight_vars = []

            # Init weights quantizers
            for name, quantizer in self._dispatcher.weight_quantizers.items():
                if is_training:
                    weight = getattr(self.layer, name).detach()
                    delattr(self.layer, name)
                    setattr(self.layer, name, weight)
                    self.register_parameter(name, torch.nn.Parameter(weight, requires_grad=True))
                else:
                    weight = getattr(self, name).detach()
                    delattr(self.layer, name)
                    setattr(self.layer, name, weight)
                quantizer.initialize_quantization(weight.shape, name, self)
                self._weight_vars.append((name, getattr(self, name), quantizer))

        def _set_activation_quantizers(self):
            """
            Initialize layer outputs and their quantizers in the wrapper
            """
            self._activation_vars = []
            for i, quantizer in enumerate(self._dispatcher.activation_quantizers):
                quantizer.initialize_quantization(None, f"tensor{i}", self)
                self._activation_vars.append(quantizer)

        def set_quantize_weights(self, quantized_weights: dict):
            """
            This function updates layer weights after quantization.

            Args:
                quantized_weights: a dict of weight to update

            Returns: None

            """
            for weight_attr in self._dispatcher.weight_quantizers.keys():
                weight = quantized_weights.get(weight_attr)
                setattr(self.layer, weight_attr, weight)

        def forward(self,
                    x: torch.Tensor,
                    *args: List[Any],
                    **kwargs: Dict[str, Any]) -> Union[torch.Tensor, List[torch.Tensor]]:
            """
            PytorchQuantizationWrapper forward functions
            Args:
                x: layer's inputs
                args: arguments to pass to internal layer.
                kwargs: key-word dictionary to pass to the internal layer.

            Returns: a tensor that simulates a quantized layer.

            """

            # ----------------------------------
            # Quantize all weights, and replace them in the underlying layer.
            # ----------------------------------
            if self._dispatcher.is_weights_quantization:

                quantized_weights = {}
                for name, unquantized_weight, quantizer in self._weight_vars:
                    s = inspect.signature(quantizer.__call__)
                    if TRAINING in s.parameters.keys():
                        quantized_weight = quantizer(unquantized_weight, self.training)
                    else:
                        quantized_weight = quantizer(unquantized_weight)

                    quantized_weights.update({name: quantized_weight})

                self.set_quantize_weights(quantized_weights)

            # ----------------------------------
            # Layer operation
            # ----------------------------------
            outputs = self.layer(x, *args, **kwargs)

            # ----------------------------------
            # Quantize all activations
            # ----------------------------------
            if self._dispatcher.is_activation_quantization:

                if not isinstance(outputs, list):
                    outputs = [outputs]

                if len(outputs) != self._dispatcher.num_act_quantizers:
                    Logger.error(f"Number of outputs {len(outputs)} is incompatible number of activation quantizers {self._dispatcher.num_act_quantizers}")  # pragma: no cover

                # Quantize all activations tensors
                outputs_quantized = []
                for quantizer, output in zip(self._activation_vars, outputs):
                    outputs_quantized.append(quantizer(output))

                outputs = outputs_quantized[0] if len(outputs_quantized) == 1 else outputs_quantized

            return outputs

else:
    class PytorchQuantizationWrapper(object):
        def __init__(self, layer, dispatcher: NodeQuantizationDispatcher):
            """
            Pytorch Quantization Wrapper takes a pytorch module and dispatcher and infer a quantized layer.

            Args:
                layer: A pytorch module.
                dispatcher: A node quantization dispatcher.
            """
            Logger.critical('Installing Pytorch is mandatory '
                            'when using PytorchQuantizationWrapper. '
                            'Could not find torch package.')  # pragma: no cover
