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

from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import \
    NoOpQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.exporter.fully_quantized.keras.builder.quantizer_to_node import \
    get_weights_quantizer_for_node, get_activations_quantizer_for_node
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.activation_quantize_config import \
    ActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.weights_activation_quantize_config \
    import \
    WeightsActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.weights_quantize_config import \
    WeightsQuantizeConfig


def get_quantization_config(node: BaseNode) -> QuantizeConfig:
    """
    Create a QuantizeConfig to wrap a layer for its corresponding node.

    Args:
        node: Node to create a QuantizeConfig for.

    Returns:
        QuantizeConfig to use for wrapping the layer from the passed node.
    """

    if node.is_weights_quantization_enabled() and not node.is_activation_quantization_enabled():
        weight_attrs = DEFAULT_KERAS_INFO.get_kernel_op_attributes(node.type)
        return WeightsQuantizeConfig(weight_attrs=weight_attrs,
                                     w_quantizer=get_weights_quantizer_for_node(node,
                                                                                weight_attrs))

    elif not node.is_weights_quantization_enabled() and node.is_activation_quantization_enabled():
        return ActivationQuantizeConfig(activation_quantizer=get_activations_quantizer_for_node(node))

    elif not node.is_weights_quantization_enabled() and not node.is_activation_quantization_enabled():
        return NoOpQuantizeConfig()

    weight_attrs = DEFAULT_KERAS_INFO.get_kernel_op_attributes(node.type)
    return WeightsActivationQuantizeConfig(activation_quantizer=get_activations_quantizer_for_node(node),
                                           w_quantizer=get_weights_quantizer_for_node(node,
                                                                                      weight_attrs),
                                           weight_attrs=DEFAULT_KERAS_INFO.get_kernel_op_attributes(node.type))
