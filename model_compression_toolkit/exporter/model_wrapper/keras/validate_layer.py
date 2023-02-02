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
from typing import Any

from keras.engine.input_layer import InputLayer

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.quantizers_infrastructure import KerasQuantizationWrapper
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import BaseInferableQuantizer



def is_keras_layer_exportable(layer: Any) -> bool:
    """
    Check whether a Keras layer is a valid exportable layer or not.

    Args:
        layer: Keras layer to check if considered to be valid for exporting.

    Returns:
        Check whether a Keras layer is a valid exportable layer or not.
    """
    # Keras Input layers are not wrapped
    if isinstance(layer, InputLayer):
        return True

    valid_layer = isinstance(layer, KerasQuantizationWrapper)
    if not valid_layer:
        Logger.error(
            f'Exportable layer must be wrapped using KerasQuantizationWrapper, but layer {layer.name} is of type '
            f'{type(layer)}') # pragma: no cover

    valid_weights_quantizers = isinstance(layer.weights_quantizers, dict)
    if not valid_weights_quantizers:
        Logger.error(
            f'KerasQuantizationWrapper must have a weights_quantizers but has a '
            f'{type(layer.weights_quantizers)} object') # pragma: no cover

    for _, weights_quantizer in layer.weights_quantizers.items():
        if not isinstance(weights_quantizer, BaseInferableQuantizer):
            Logger.error(
                f'weights_quantizer must be a BaseInferableQuantizer object but has a '
                f'{type(weights_quantizer)} object')  # pragma: no cover

    valid_activation_quantizers = isinstance(layer.activation_quantizers, list)
    if not valid_activation_quantizers:
        Logger.error(
            f'KerasQuantizationWrapper must have a activation_quantizers list but has a '
            f'{type(layer.activation_quantizers)} object') # pragma: no cover

    for activation_quantizers in layer.activation_quantizers:
        if not isinstance(activation_quantizers, BaseInferableQuantizer):
            Logger.error(
                f'activation_quantizers must be a BaseInferableQuantizer object but has a '
                f'{type(activation_quantizers)} object')  # pragma: no cover

    quantizers = layer.activation_quantizers + list(layer.weights_quantizers.values())
    is_valid_quantizers = all([isinstance(x, BaseInferableQuantizer) for x in quantizers])
    if not is_valid_quantizers:
        Logger.error(f'Found a quantizer that is not of type BaseInferableQuantizer') # pragma: no cover

    return True
