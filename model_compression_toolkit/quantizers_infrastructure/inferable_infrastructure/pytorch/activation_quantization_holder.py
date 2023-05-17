# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.constants import FOUND_TORCH
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.quantizers_infrastructure import BaseInferableQuantizer

if FOUND_TORCH:
    import torch

    class PytorchActivationQuantizationHolder(torch.nn.Module):
        """
        Pytorch module to hold an activation quantizer and quantize during inference.
        """
        def __init__(self,
                     activation_holder_quantizer: BaseInferableQuantizer,
                     **kwargs):
            """

            Args:
                activation_holder_quantizer: Quantizer to use during inference.
                **kwargs: Key-word arguments for the base layer
            """

            super(PytorchActivationQuantizationHolder, self).__init__(**kwargs)
            self.activation_holder_quantizer = activation_holder_quantizer

        def forward(self, inputs):
            """
            Quantizes the input tensor using the activation quantizer of class PytorchActivationQuantizationHolder.

            Args:
                inputs: Input tensors to quantize with the activation quantizer.

            Returns: Output of the activation quantizer (quantized input tensor).

            """
            return self.activation_holder_quantizer(inputs)

else:
    class PytorchActivationQuantizationHolder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.critical('Installing Pytorch is mandatory '
                            'when using PytorchActivationQuantizationHolder. '
                            'Could not find the torch package.')  # pragma: no cover