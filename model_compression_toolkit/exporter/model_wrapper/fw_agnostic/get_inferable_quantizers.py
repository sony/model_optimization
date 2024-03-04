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
from typing import Dict, List, Tuple, Callable
from model_compression_toolkit.core.common import BaseNode


def get_inferable_quantizers(node: BaseNode,
                             get_weights_quantizer_for_node: Callable,
                             get_activations_quantizer_for_node: Callable,
                             attributes_names: List[str] = []) -> Tuple[Dict, List]:
    """
    Create quantizers to wrap a layer for its corresponding node.

    Args:
        node: Node to create quantizers for.
        get_weights_quantizer_for_node: A function that returns weights quantizer for the node attributes.
        get_activations_quantizer_for_node: A function that returns activation quantizer for the node activation tensor.
        attributes_names: A potential list of attribute names to set weights quantizers to.

    Returns:
        weight_quantizers: A dictionary between a weight's name to its quantizer.
        activation_quantizers: A list of activations quantization, one for each layer output.
    """

    weight_quantizers = {}
    activation_quantizers = []

    for attr in attributes_names:
        if node.is_weights_quantization_enabled(attr):
            weight_quantizer = get_weights_quantizer_for_node(node, attr)
            weight_quantizers[attr] = weight_quantizer

    if node.is_activation_quantization_enabled():
        num_of_outputs = len(node.output_shape) if isinstance(node.output_shape, list) else 1
        activation_quantizers = [get_activations_quantizer_for_node(node)] * num_of_outputs

    return weight_quantizers, activation_quantizers
