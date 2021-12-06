# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
import copy

from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common import Graph, BaseNode

def apply_bias_correction_to_graph(graph_to_apply_bias_correction: Graph,
                                   fw_info: FrameworkInfo,
                                   fw_impl: FrameworkImplementation) -> Graph:
    """
    Get a graph, where each node has a final weights quantization configuration (with a bias
    correction term in it), and apply the bias correction for each node in the graph.

    Args:
        graph_to_apply_bias_correction: Graph to apply bias correction to.
        fw_info: Framework information (e.g, operators to quantize their weights).
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Graph with bias correction apply to its' nodes.
    """

    graph = copy.deepcopy(graph_to_apply_bias_correction)
    for n in graph.nodes:
        if fw_info.in_kernel_ops(n) and n.final_weights_quantization_cfg.enable_weights_quantization:
            # If a kernel was quantized and weights bias correction is enabled in n.quantization_cfg,
            # a bias correction term was calculated during model preparation, and is used now in the node's bias term.
            if n.final_weights_quantization_cfg.weights_bias_correction:
                _apply_bias_correction_to_node(n, fw_impl)
    return graph


def _apply_bias_correction_to_node(node:BaseNode,
                                   fw_impl: FrameworkImplementation):
    """
    Set new bias to node using the bias-correction term that is stored in the
    final weights quantization configuration.

    Args:
        node: Node to set its corrected bias after bias-correction.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    """
    correction = node.final_weights_quantization_cfg.bias_corrected

    bias = node.get_weights_by_keys(fw_impl.constants.BIAS)  # get original bias from node's weights

    if bias is not None:  # It the layer has bias, we subtract the correction from original bias
        node.set_weights_by_keys(fw_impl.constants.BIAS, node.get_weights_by_keys(fw_impl.constants.BIAS) - correction)

    else:  # It the layer has no bias, we consider it as if it has and its value is 0.
        node.set_weights_by_keys(fw_impl.constants.BIAS, - correction)
        node.framework_attr[fw_impl.constants.USE_BIAS] = True  # Mark the use_bias attribute of the node.
