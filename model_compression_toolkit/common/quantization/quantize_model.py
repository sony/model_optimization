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


from typing import List, Tuple, Dict

import numpy as np

from model_compression_toolkit import common
from model_compression_toolkit.common import threshold_selection, Node
from model_compression_toolkit.common.defaultdict import DefaultDict
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.statistics_collector import BaseStatsContainer
from model_compression_toolkit.keras.constants import KERNEL, BIAS, USE_BIAS

# If the quantization config does not contain kernel channel mapping or the weights
# quantization is not per-channel, we use a dummy channel mapping.
dummy_channel_mapping = DefaultDict({}, lambda: (None, None))


def get_activations_params(n: Node,
                           graph: Graph) -> Tuple[Dict[str, float], bool]:
    """
    Compute the activations params for a given node in a graph according to a params function.

    Args:
        n: Node to compute its' activations threshold.
        graph: Graph the node is in.

    Returns:
        Tuple of the computed quantization params and sign for the node's activations quantization.
    """
    out_stats_container = graph.get_out_stats_collector(n)
    bins_values, bins_counts = None, None

    # If the statistics container collected the histogram, we start by filtering outliers using z threshold
    # filtering, and then computing the threshold based on the filtered histogram.
    if out_stats_container.collect_histogram:
        bins_values, bins_counts = out_stats_container.hc.get_histogram()
        bins_counts = threshold_selection.z_score_filter(n.activation_quantization_cfg.z_threshold,
                                                         bins_values,
                                                         bins_counts)
    min_value, max_value = out_stats_container.get_min_max_values()

    if out_stats_container.use_min_max:
        signed = min_value < 0
    else:
        signed = np.any(bins_values < 0)

    activation_params = n.activation_quantization_cfg.activation_quantization_params_fn(bins_values,
                                                                                        bins_counts,
                                                                                        n.activation_quantization_cfg.l_p_value,
                                                                                        n.activation_quantization_cfg.activation_n_bits,
                                                                                        min_value,
                                                                                        max_value,
                                                                                        min_threshold=n.activation_quantization_cfg.min_threshold)

    return activation_params, signed


def compute_bias_correction(kernel: np.ndarray,
                            quantized_kernel: np.ndarray,
                            in_statistics_container: BaseStatsContainer,
                            output_channels_axis: int,
                            input_channels: int):
    """
    Compute the bias correction term for the bias in the error on the layer’s output,
    that is introduced by the weights quantization.
    For more info: https://arxiv.org/abs/1906.04721
    
    Args:
        kernel: Float kernel of the layer that its output is biased.
        quantized_kernel: Quantized kernel of the layer that its output is biased.
        in_statistics_container: Inputs statistics of the quantized layer that has the bias error.
        output_channels_axis: Output channels index of the given kernel.
        input_channels: Input channels index of the given kernel.

    Returns:
        Term to add to the bias of the quantized layer in order to correct the expected
        bias due to weights quantization.
    """

    correction_term = None
    quantization_error = quantized_kernel - kernel
    mu = in_statistics_container.get_mean()
    eps = np.sum(quantization_error,
                 axis=tuple([i for i in range(len(quantization_error.shape)) if
                             i not in [output_channels_axis, input_channels]]))

    if output_channels_axis == input_channels:
        correction_term = mu * eps

    else:
        if output_channels_axis > input_channels:
            eps = np.transpose(eps)
        correction_term = np.matmul(eps, mu)

    return correction_term


def calculate_quantization_params(graph: Graph,
                                  fw_info: FrameworkInfo,
                                  nodes: List[Node] = [],
                                  specific_nodes: bool = False):
    """
    For a graph, go over its nodes, compute quantization params (for both weights and activations according
    to the given framework info), and create and attach a NodeQuantizationConfig to each node (containing the
    computed params).
    By default, the function goes over all nodes in the graph. However, the specific_nodes flag enables
    to compute quantization paramss for specific nodes if the default behavior is unnecessary. For that,
    a list of nodes nodes should be passed as well.

    Args:
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        graph: Graph to compute its nodes' thresholds.
        nodes: List of nodes to compute their thresholds instead of computing it for all nodes in the graph.
        specific_nodes: Flag to compute thresholds for only specific nodes.

    """

    # Create a list of nodes to compute their thresholds
    nodes_list: List[common.Node] = nodes if specific_nodes else graph.nodes()

    for n in nodes_list:  # iterate only nodes that we should compute their thresholds

        weights_params, activation_params, activation_is_signed, output_channels_axis, \
        input_channels_axis, activation_threshold_float = {}, {}, None, None, None, None

        if fw_info.in_kernel_ops(n):  # If the node has a kernel to quantize

            # If weights should be quantized per-channel but a kernel channels mapping is missing.
            if n.weights_quantization_cfg.weights_per_channel_threshold and \
                    fw_info.kernel_channels_mapping is None:
                common.Logger.warning('Weights Per Channel Quantization requires channel mapping function,'
                                      ' but framework info does not contain one')

            # Set a kernel channels mapping
            if n.weights_quantization_cfg.weights_per_channel_threshold:
                kernel_channels_mapping = fw_info.kernel_channels_mapping
            else:  # If kernel mapping is missing, we use a dummy channels mapping
                kernel_channels_mapping = dummy_channel_mapping

            kernel = n.get_weights_by_keys(KERNEL)
            output_channels_axis, input_channels_axis = kernel_channels_mapping.get(n.layer_class)
            if n.weights_quantization_cfg is not None and n.weights_quantization_cfg.weights_quantization_params_fn \
                    is not None:
                weights_params = n.weights_quantization_cfg.weights_quantization_params_fn(kernel,
                                                                                           p=n.weights_quantization_cfg.l_p_value,
                                                                                           n_bits=n.weights_quantization_cfg.weights_n_bits,
                                                                                           per_channel=n.weights_quantization_cfg.weights_per_channel_threshold and output_channels_axis is not None,
                                                                                           channel_axis=output_channels_axis,
                                                                                           min_threshold=n.weights_quantization_cfg.min_threshold)
            else:
                weights_params = {}
            if n.output_quantization:  # If node's activations should be quantized as well, we compute its
                # activation threshold
                activation_params, activation_is_signed = get_activations_params(n=n,
                                                                                 graph=graph)

        elif fw_info.in_activation_ops(n):  # If node has no kernel, but its activations should be quantized
            if n.output_quantization:
                activation_params, activation_is_signed = get_activations_params(n=n,
                                                                                 graph=graph)
        # If node should not be quantized at all
        elif fw_info.in_no_quantization_ops(n):
            pass  # pragma: no cover

        # If layer type is not in framework info, log a warning.
        else:
            common.Logger.warning(f"Warning: unknown layer: {n.layer_class.__name__}")

        # Create a NodeQuantizationConfig containing all quantization params and attach it to the node
        if n.activation_quantization_cfg is not None:
            n.activation_quantization_cfg.set_activation_quantization_param(activation_params)
            n.activation_quantization_cfg.activation_is_signed = activation_is_signed

        if n.weights_quantization_cfg is not None:
            n.weights_quantization_cfg.set_weights_quantization_param(weights_params)
            n.weights_quantization_cfg.weights_channels_axis = output_channels_axis


def quantize_model(graph: Graph,
                   fw_info: FrameworkInfo):
    """
    Get a graph representing a model, and quantize its nodes' weights.
    Each node is quantized according to the passed framework info and quantization configuration.
    If weights bias correction is enabled in the quantization configuration, a bias correction term
    is calculated and subtracted from the original node's bias. The graph is quantized in-place.

    Args:
        graph: Graph to quantize its nodes.
        fw_info: Framework information needed for quantizing the graph's nodes' weights and activations.

    """
    # Iterate over nodes in the graph and quantize each node's weights and activations
    # (according to operators groups in framework info).
    for n in graph.nodes():

        if fw_info.in_kernel_ops(n) and n.weights_quantization_cfg.enable_weights_quantization:
            # If weights should be quantized per-channel but a kernel channels mapping is missing.
            if n.weights_quantization_cfg.weights_per_channel_threshold and fw_info.kernel_channels_mapping is None:
                common.Logger.warning(
                    'Weights Per Channel Quantization requires channel mapping function but framework info '
                    'does not contain one')

            # Use kernel channels mapping if exist (otherwise, use a dummy mapping)
            if not n.weights_quantization_cfg.weights_per_channel_threshold and not \
                    n.weights_quantization_cfg.weights_bias_correction:
                kernel_channels_mapping = dummy_channel_mapping
            else:
                kernel_channels_mapping = fw_info.kernel_channels_mapping

            kernel = n.get_weights_by_keys(KERNEL)
            output_channels_axis, input_channels_axis = kernel_channels_mapping.get(n.layer_class)

            # if n.quantization_cfg is None:
            #     common.Logger.error(f"Can not find quantization config for node name: {n.name}")

            quantized_kernel = n.weights_quantization_cfg.weights_quantization_fn(kernel,
                                                                                  n_bits=n.weights_quantization_cfg.weights_n_bits,
                                                                                  signed=True,
                                                                                  quantization_params=n.weights_quantization_cfg.weights_quantization_params,
                                                                                  per_channel=n.weights_quantization_cfg.weights_per_channel_threshold,
                                                                                  output_channels_axis=output_channels_axis)

            # If a kernel was quantized and weights bias correction is enabled in n.quantization_cfg,
            # a bias correction term is being calculated and used in the node's bias term.
            if n.weights_quantization_cfg.weights_bias_correction:
                if output_channels_axis is None:
                    common.Logger.error(
                        f'Unknown output channel axis for node named: {n.name},'
                        f' please update channel mapping function')
                if input_channels_axis is None:
                    common.Logger.error(
                        f'Unknown input channel axis for node named: {n.name},'
                        f' please update channel mapping function')

                # Compute the bias correction term.
                correction = compute_bias_correction(kernel,
                                                     quantized_kernel,
                                                     graph.get_in_stats_collector(n),
                                                     output_channels_axis,
                                                     input_channels_axis)

                bias = n.get_weights_by_keys(BIAS)  # get original bias from node's weights

                if bias is not None:  # It the layer has bias, we subtract the correction from original bias
                    n.set_weights_by_keys(BIAS, n.get_weights_by_keys(BIAS) - correction)

                else:  # It the layer has no bias, we consider it as if it has and its value is 0.
                    n.set_weights_by_keys(BIAS, - correction)
                    n.framework_attr[USE_BIAS] = True  # Mark the use_bias attribute of the node.

            common.Logger.debug(
                f'Node name: {n.name} has the following quantization params: '
                f'{str(n.weights_quantization_cfg.weights_quantization_params)}')

            # Set the kernel node to be the quantized kernel.
            n.set_weights_by_keys(KERNEL, quantized_kernel)
