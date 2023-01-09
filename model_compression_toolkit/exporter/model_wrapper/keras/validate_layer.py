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
from model_compression_toolkit.quantizers_infrastructure import KerasQuantizationWrapper, \
    KerasNodeQuantizationDispatcher
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
            f'{type(layer)}')

    valid_dispatcher = isinstance(layer.dispatcher, KerasNodeQuantizationDispatcher)
    if not valid_dispatcher:
        Logger.error(
            f'KerasQuantizationWrapper must have a dispatcher of type KerasNodeQuantizationDispatcher but has a '
            f'{type(layer.dispatcher)} object as a dispatcher')

    dispatcher_quantizers = layer.dispatcher.activation_quantizers + list(layer.dispatcher.weight_quantizers.values())
    inferable_quantizers = all([isinstance(x, BaseInferableQuantizer) for x in dispatcher_quantizers])
    if not inferable_quantizers:
        Logger.error(f'Found a quantizer in the dispatcher that is not of type BaseInferableQuantizer')

    return True
