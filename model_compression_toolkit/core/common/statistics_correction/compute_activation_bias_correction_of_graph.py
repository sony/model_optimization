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
import numpy as np
from typing import Any, Callable

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo


def get_previous_node_with_activation_quantization(linear_node: BaseNode,
                                                   graph: Graph) -> Any:
    """
    Search recursively for the previous node with activation quantization.

    Args:
        linear_node: Node to search for its previous node.
        graph: Graph the node is in.

    Returns:
        The previous node (if found) or None if it was not found or there are multiple incoming edges to one of
        nodes during the search (which means, the substitution can not be applied).
    """

    prev_nodes = graph.get_prev_nodes(linear_node)

    if len(prev_nodes) != 1:
        return None  # pragma: no cover

    prev_node = prev_nodes[0]

    activation_quantization_config = prev_node.final_activation_quantization_cfg

    # Search for node with activation quantization
    if activation_quantization_config.enable_activation_quantization:
        return prev_node
    else:
        return get_previous_node_with_activation_quantization(prev_node, graph)


def calculate_bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    """
    Calculate the centers of bins given their edges.

    Args:
        bin_edges: Array of bin edges.

    Returns:
        np.ndarray: Array of bin centers.
    """
    # Calculate the centers by averaging continuous bin edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return bin_centers


def compute_activation_bias_correction(graph: Graph,
                                       quant_config: QuantizationConfig,
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
        quant_config: QuantizationConfig of how the model should be quantized.
        fw_info: Framework info like lists of nodes their kernel should quantized.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        linear_node: Node to compute the activation bias correction for.
        prev_node: Node to compute the activation error caused by his activation quantization.
        kernel_size: The framework specific attribute name of the convolution layer's kernel size.

    Returns:
        Graph with activation bias correction term for each node.
    """

    # Retrieve the 'kernel_size' value if it exists and ensure it is None, 1, or (1, 1). This feature supports only
    # Dense/Linear layers and convolution layers with kernel size  of 1 or (1, 1).
    # For Dense/Linear layers, which lack a 'kernel_size' attribute, the result will be None, and no restriction
    # applies in that case.
    if linear_node.framework_attr.get(kernel_size) not in [None, 1, (1, 1)]:
        # If the kernel size is not 1 or (1, 1), return the current graph unmodified
        return graph

    prev_node_act_quant_cfg = prev_node.final_activation_quantization_cfg

    # Check if the previous node's has activation quantization configuration and if the previous node have the
    # histogram collector.
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

    # Calculate the normalized bias as a percentage of the float center norm
    float_centers_norm1 = np.abs(mean_float_centers)
    normalized_bias = 100 * np.abs(mean_diff) / float_centers_norm1

    # If the normalized bias is below the activation bias correction threshold, return the graph unmodified.
    # By default, the threshold is set to 0, allowing all nodes to proceed in this case.
    if normalized_bias < quant_config.activation_bias_correction_threshold:
        return graph

    kernel = linear_node.get_weights_by_keys(fw_info.kernel_ops_attributes_mapping.get(linear_node.type)[0])

    # Compute the activation bias correction by applying the quantization error to the kernel, resulting in an output
    # size matching the number of output channels.
    if kernel is not None:

        # Get the axes that are not the output channel.
        output_channel_index, input_channel_index = fw_info.kernel_channels_mapping.get(linear_node.type)
        axis_not_output_channel = list(range(len(kernel.shape)))
        axis_not_output_channel.remove(output_channel_index)

        # Special case of depthwise_conv2d in tensorflow, where we have a depth multiplier for the filters.
        if output_channel_index == input_channel_index:
            axis_not_output_channel.remove(3)  # 3 is the depth multiplier index.

        activation_bias_correction_term = mean_diff * np.sum(kernel, axis=tuple(axis_not_output_channel))
        linear_node.final_activation_quantization_cfg.activation_bias_correction_term = (
            activation_bias_correction_term.flatten())
    return graph


def compute_activation_bias_correction_of_graph(graph: Graph,
                                                quant_config: QuantizationConfig,
                                                fw_info: FrameworkInfo,
                                                fw_impl: FrameworkImplementation,
                                                activation_bias_correction_node_matchers: Callable,
                                                kernel_size: str) -> Graph:
    """
    Compute the activation bias correction term for the graph.

    Args:
        graph: Graph with nodes to compute the activation bias correction.
        quant_config: QuantizationConfig of how the model should be quantized.
        fw_info: Framework info like lists of nodes their kernel should quantized.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        activation_bias_correction_node_matchers: Function to match the layers for activation bias correction.
        kernel_size: The framework specific attribute name of the convolution layer's kernel size.


    Returns:
        Graph with activation bias correction term for each relevant node.
    """
    linear_node_types = activation_bias_correction_node_matchers()

    for n in graph.nodes:
        if linear_node_types.apply(n):
            prev_node = get_previous_node_with_activation_quantization(n, graph)
            if prev_node is not None:
                graph = compute_activation_bias_correction(graph=graph,
                                                           quant_config=quant_config,
                                                           fw_info=fw_info,
                                                           fw_impl=fw_impl,
                                                           linear_node=n,
                                                           prev_node=prev_node,
                                                           kernel_size=kernel_size)
    return graph
