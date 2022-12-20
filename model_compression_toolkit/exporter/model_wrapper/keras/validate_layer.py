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
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapperV2

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.exporter.model_wrapper.keras.builder.quantize_config_to_node import \
    SUPPORTED_QUANTIZATION_CONFIG
from model_compression_toolkit.exporter.model_wrapper.keras.extended_quantize_wrapper import ExtendedQuantizeWrapper


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

    valid_layer = isinstance(layer, ExtendedQuantizeWrapper)
    if not valid_layer:
        Logger.error(f'Exportable layer must be wrapped using ExtendedQuantizeWrapper, but layer {layer.name} is of type {type(layer)}')

    valid_quantize_config = type(layer.quantize_config) in SUPPORTED_QUANTIZATION_CONFIG
    if not valid_quantize_config:
        Logger.error(f'QuantizeConfig of layer is not supported. Type: {type(layer.quantize_config)}. Supported configs: {SUPPORTED_QUANTIZATION_CONFIG}.')

    return True
