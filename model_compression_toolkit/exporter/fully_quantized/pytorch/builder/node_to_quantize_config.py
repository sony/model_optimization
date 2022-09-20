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
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.pytorch.back2framework.quantization_wrapper.wrapper_quantize_config import \
    WrapperQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.pytorch.builder.node_to_quantizer import \
    get_weights_quantizer_for_node, \
    get_activations_quantizer_for_node
from model_compression_toolkit.exporter.fully_quantized.pytorch.wrappers_quantize_configs.activation_quantize_config \
    import \
    ActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.pytorch.wrappers_quantize_configs \
    .no_quantization_quantize_config import \
    NoQuantizationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.pytorch.wrappers_quantize_configs \
    .weights_activation_quantize_config import \
    WeightsActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.pytorch.wrappers_quantize_configs.weights_quantize_config \
    import \
    WeightsQuantizeConfig


def get_quantization_config(node: BaseNode) -> WrapperQuantizeConfig:
    """
    Create a WrapperQuantizeConfig to wrap a layer for its corresponding node.

    Args:
        node: Node to create a WrapperQuantizeConfig for.

    Returns:
        WrapperQuantizeConfig to use for wrapping the layer from the passed node.

    """

    if node.is_activation_quantization_enabled() and node.is_weights_quantization_enabled():
        weight_quantizers = get_weights_quantizer_for_node(node)
        activation_quantizers = get_activations_quantizer_for_node(node)
        return WeightsActivationQuantizeConfig(weight_quantizers=weight_quantizers,
                                               activation_quantizers=activation_quantizers)

    elif not node.is_weights_quantization_enabled() and node.is_activation_quantization_enabled():
        activation_quantizers = get_activations_quantizer_for_node(node)
        return ActivationQuantizeConfig(activation_quantizers=activation_quantizers)

    elif not node.is_weights_quantization_enabled() and not node.is_activation_quantization_enabled():
        return NoQuantizationQuantizeConfig()

    weight_quantizers = get_weights_quantizer_for_node(node)
    return WeightsQuantizeConfig(weight_quantizers=weight_quantizers)
