# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List, Tuple, Any, Callable

import numpy as np

from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher


def get_next_nodes_to_correct(node: BaseNode,
                              graph: Graph,
                              linear_node_types: NodeOperationMatcher,
                              bypass_node_types: NodeOperationMatcher,
                              bypass_nodes: List = None) -> Tuple[Any, Any]:
    """
    Search for the previous node which is not a bypass node of a given node. Go over the previous nodes of the node
    and recursively search for a node.

    Args:
        node: Node to search for its previous node.
        graph: Graph the node is in.
        linear_node_types: Types of linear nodes to consider.
        bypass_node_types: Types of nodes for bypassing to consider.
        bypass_nodes: a list of bypass nodes found while running this function

    Returns: The previous node (if found) and a list of bypass nodes (if any), or Nones if it were not found or there
    are multiple incoming edges to one of nodes during the search (which means, the substitution can not be applied).
    """

    prev_nodes = graph.get_prev_nodes(node)

    if len(prev_nodes) != 1:
        return None, None  # pragma: no cover

    prev_node = prev_nodes[0]

    # If the previous node is not a bypass type, return it as the valid node along with any bypass nodes
    if not bypass_node_types.apply(prev_node):
        return prev_node, bypass_nodes

    # If the previous node is a bypass node type, add it to the bypass_nodes list and continue searching
    if bypass_node_types.apply(prev_node):
        if bypass_nodes:
            bypass_nodes.append(prev_node)
        else:
            bypass_nodes = [prev_node]
        return get_next_nodes_to_correct(node=prev_node,
                                         graph=graph,
                                         linear_node_types=linear_node_types,
                                         bypass_node_types=bypass_node_types,
                                         bypass_nodes=bypass_nodes)
    return None, None  # pragma: no cover


def calculate_bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    """
    Calculate the centers of bins given their edges.

    Parameters:
    bin_edges: Array of bin edges.

    Returns:
    np.ndarray: Array of bin centers.
    """
    # Ensure bin_edges is a numpy array
    bin_edges = np.array(bin_edges, dtype=np.float32)

    # Calculate the centers by averaging continuous bin edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return bin_centers


def compute_activation_bias_correction(graph: Graph,
                                       core_config: CoreConfig,
                                       fw_info: FrameworkInfo,
                                       fw_impl: FrameworkImplementation,
                                       linear_node: BaseNode,
                                       prev_node: BaseNode,
                                       kernel_size: str) -> Graph:
    """
    Compute the activation bias correction term, and store it in the final activation
    quantization configuration.

    Args:
        graph: Graph with nodes to compute the activation bias correction for each node's final activation quantization configuration.
        core_config: Configuration object containing parameters of how the model should be quantized.
        fw_info: Framework info like lists of nodes their kernel should quantized.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        linear_node: Node to compute the activation bias correction for.
        prev_node: Node to compute the activation error caused by his activation quantization.
        kernel_size: The framework specific attribute name of the convolution layer's kernel size.

    Returns:
        Graph with activation bias correction term for each node.
    """

    # Check if 'kernel_size' is a key in the framework-specific attributes of the linear_node, if it is then the
    # linear_node is a convolution
    if kernel_size in linear_node.framework_attr.keys():
        # Retrieve the value of 'kernel_size' and check if it is not 1 or (1, 1). This feature supports only kernel
        # size of 1 or (1, 1).
        if linear_node.framework_attr.get(kernel_size) not in [1, (1, 1)]:
            # If the kernel size is not 1 or (1, 1), return the current graph unmodified
            return graph

    prev_node_act_quant_cfg = prev_node.final_activation_quantization_cfg

    # Check if the previous node's has activation quantization configuration and if the previous node have the
    # histogram collector
    if prev_node_act_quant_cfg is None or not hasattr(graph.get_out_stats_collector(prev_node), 'hc'):
        return graph  # pragma: no cover

    float_bins, float_count = graph.get_out_stats_collector(prev_node).hc.get_histogram()

    # Calculate the centers of the float bins
    float_centers = calculate_bin_centers(float_bins)

    # Quantize the bin edges and calculate the centers of the quantized bins
    quant_bins = prev_node_act_quant_cfg.quantize_node_output(fw_impl.to_tensor(float_bins))
    quant_bins = fw_impl.to_numpy(quant_bins)
    quant_centers = calculate_bin_centers(quant_bins)

    # Calculate the mean of the both the float and the quantized bin centers, weighted by the bin counts
    mean_float_centers = np.sum(float_centers * float_count) / np.sum(float_count)
    mean_quant_centers = np.sum(quant_centers * float_count) / np.sum(float_count)

    # Compute the difference between the mean quantized center and the mean float center
    mean_diff = mean_quant_centers - mean_float_centers

    # Check if activation bias correction is enabled based on the configured threshold
    if core_config.quantization_config.activation_bias_correction_threshold > 0:

        # Calculate the normalized bias as a percentage of the float center norm
        float_centers_norm1 = np.abs(mean_float_centers)
        normalized_bias = 100 * np.abs(mean_diff) / float_centers_norm1

        # If the normalized bias is below the activation bias correction threshold, return the unmodified graph
        if normalized_bias < core_config.quantization_config.activation_bias_correction_threshold:
            return graph

    # The correction term is a function of the layer type.
    kernel = linear_node.get_weights_by_keys(fw_info.kernel_ops_attributes_mapping.get(linear_node.type)[0])

    if kernel is not None:
        output_channel_index, input_channel_index = fw_info.kernel_channels_mapping.get(linear_node.type)
        axis_not_output_channel = list(range(len(kernel.shape)))
        axis_not_output_channel.remove(output_channel_index)

        if output_channel_index == input_channel_index:
            axis_not_output_channel.remove(3)  # 3 is the depth multiplier index

        activation_bias_correction_term = mean_diff * np.sum(kernel, axis=tuple(axis_not_output_channel))
        linear_node.final_activation_quantization_cfg.activation_bias_correction_term = activation_bias_correction_term.flatten()
    return graph


def compute_activation_bias_correction_of_graph(graph: Graph,
                                                core_config: CoreConfig,
                                                fw_info: FrameworkInfo,
                                                fw_impl: FrameworkImplementation,
                                                activation_bias_correction_node_matchers: Callable,
                                                kernel_size: str) -> Graph:
    """
    Compute the activation bias correction term for the graph.

    Args:
        graph: Graph with nodes to compute the activation bias correction.
        core_config: Configuration object containing parameters of how the model should be quantized.
        fw_info: Framework info like lists of nodes their kernel should quantized.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        activation_bias_correction_node_matchers: Function to match the layers for activation bias correction.
        kernel_size: The framework specific attribute name of the convolution layer's kernel size.


    Returns:
        Graph with activation bias correction term for each relevant node.
    """
    linear_node_types, bypass_node_types = activation_bias_correction_node_matchers()

    for n in graph.nodes:
        if linear_node_types.apply(n):
            prev_node, _ = get_next_nodes_to_correct(node=n,
                                                     graph=graph,
                                                     linear_node_types=linear_node_types,
                                                     bypass_node_types=bypass_node_types)
            if prev_node is not None:
                graph = compute_activation_bias_correction(graph=graph,
                                                           core_config=core_config,
                                                           fw_info=fw_info,
                                                           fw_impl=fw_impl,
                                                           linear_node=n,
                                                           prev_node=prev_node,
                                                           kernel_size=kernel_size)
    return graph
