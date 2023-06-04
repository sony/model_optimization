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
from typing import Any

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import FOUND_TORCH


if FOUND_TORCH:
    import torch.nn as nn
    from mct_quantizers import PytorchQuantizationWrapper
    from mct_quantizers.pytorch.quantizers import BasePyTorchInferableQuantizer
    from mct_quantizers.pytorch.activation_quantization_holder import PytorchActivationQuantizationHolder

    def is_pytorch_layer_exportable(layer: Any) -> bool:
        """
        Check whether a torch Module is a valid exportable module or not.

        Args:
            layer: PyTorch module to check if considered to be valid for exporting.

        Returns:
            Check whether a PyTorch layer is a valid exportable layer or not.
        """
        if not isinstance(layer, nn.Module):
            Logger.error(f'Exportable layer must be a nn.Module layer, but layer {layer.name} is of type {type(layer)}') # pragma: no cover

        if isinstance(layer, PytorchQuantizationWrapper):
            valid_weights_quantizers = isinstance(layer.weights_quantizers, dict)
            if not valid_weights_quantizers:
                Logger.error(
                    f'PytorchQuantizationWrapper must have a weights_quantizers but has a '
                    f'{type(layer.weights_quantizers)} object') # pragma: no cover

            if len(layer.weights_quantizers) == 0:
                Logger.error(f'PytorchQuantizationWrapper must have at least one weight quantizer, but found {len(layer.weights_quantizers)} quantizers.'
                             f'If layer is not quantized it should be a Keras layer.')

            for _, weights_quantizer in layer.weights_quantizers.items():
                if not isinstance(weights_quantizer, BasePyTorchInferableQuantizer):
                    Logger.error(
                        f'weights_quantizer must be a BasePyTorchInferableQuantizer object but has a '
                        f'{type(weights_quantizer)} object')  # pragma: no cover

        elif isinstance(layer, PytorchActivationQuantizationHolder):
            if not isinstance(layer.activation_holder_quantizer, BasePyTorchInferableQuantizer):
                Logger.error(
                    f'activation quantizer in PytorchActivationQuantizationHolder'
                    f' must be a BasePyTorchInferableQuantizer object but has a '
                    f'{type(layer.activation_holder_quantizer)} object')  # pragma: no cover

        return True

else:
    def is_pytorch_layer_exportable(*args, **kwargs):  # pragma: no cover
        Logger.error('Installing torch is mandatory '
                     'when using is_pytorch_layer_exportable. '
                     'Could not find PyTorch package.')