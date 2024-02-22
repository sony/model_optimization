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
from typing import Dict, List, Tuple
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.exporter.model_wrapper.keras.builder.node_to_quantizer import \
    get_weights_quantizer_for_node, get_activations_quantizer_for_node


def _extract_keras_attr_name(attr_name: str) -> str:
    """
    Keras weight attributes names appear in a certain patters - "layer_type_name/attribute_variable_name:idx".
    In order to map between a layer attribute to its quantizer, we need to extract the actual attribute name.
    E.g., "conv2d/kernel:0" --> "kernel".

    Args:
        attr_name: A Keras attribute name.

    Returns: A decomposed attribute name.
    """

    clean_name = attr_name.split('/')[-1]
    clean_name = clean_name.split(":")[0]

    return clean_name


def get_quantization_quantizers(node: BaseNode) -> Tuple[Dict, List]:
    """
    Create quantizers to wrap a layer for its corresponding node.

    Args:
        node: Node to create quantizers for.

    Returns:
        weight_quantizers: A dictionary between a weight's name to its quantizer.
        activation_quantizers: A list of activations quantization, one for each layer output.
    """
    weight_quantizers = {}
    activation_quantizers = []

    for attr in node.get_node_weights_attributes():
        if node.is_weights_quantization_enabled(attr):
            weight_quantizer = get_weights_quantizer_for_node(node, attr)
            # var_attr_name = _extract_keras_attr_name(attr)
            # weight_quantizers[var_attr_name] = weight_quantizer
            weight_quantizers[attr] = weight_quantizer

    if node.is_activation_quantization_enabled():
        num_of_outputs = len(node.output_shape) if isinstance(node.output_shape, list) else 1
        activation_quantizers = [get_activations_quantizer_for_node(node)] * num_of_outputs

    return weight_quantizers, activation_quantizers
