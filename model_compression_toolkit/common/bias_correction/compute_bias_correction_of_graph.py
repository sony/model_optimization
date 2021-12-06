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
from typing import Any

import numpy as np

from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common import BaseNode, Logger, Graph
from model_compression_toolkit.common.quantization.quantize_node import get_quantized_kernel_by_weights_qc
from model_compression_toolkit.common.statistics_collector import BaseStatsContainer


def compute_bias_correction_of_graph(graph_co_compute_bias: Graph,
                                     fw_info: FrameworkInfo,
                                     fw_impl: FrameworkImplementation) -> Graph:
    """
    For each node in a graph, and for each candidate weights quantization configuration,
    compute the bias-correction term, and store it in the candidate weights quantization configuration.

    Args:
        graph_co_compute_bias: Graph with nodes to compute the bias correction for
        each node's weights quantization configuration candidates.
        fw_info: Framework info like lists of nodes their kernel should quantized.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Graph with bias correction for each weights quantization configuration candidate
        for each node.
    """

    graph = copy.deepcopy(graph_co_compute_bias)
    for n in graph.nodes:
        if fw_info.in_kernel_ops(n):
            _compute_bias_correction_per_candidate_qc(n,
                                                      fw_info,
                                                      graph.get_in_stats_collector(n),
                                                      fw_impl=fw_impl)
    return graph


def _compute_bias_correction_per_candidate_qc(node: BaseNode,
                                              fw_info: FrameworkInfo,
                                              node_in_stats_collector: BaseStatsContainer,
                                              fw_impl: FrameworkImplementation):
    """
    For each candidate weights quantization configuration of a given node,
    compute the bias-correction term, and store it in the candidate weights quantization configuration.

    Args:
        node: Node to compute the bias correction for its different candidates.
        fw_info: Framework info like lists of nodes their kernel should quantized.
        node_in_stats_collector: Statistics collector of the node for the mean per-channel.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    """

    if node.candidates_weights_quantization_cfg is not None:
        for weights_qc in node.candidates_weights_quantization_cfg:
            if fw_info.in_kernel_ops(node) and weights_qc.enable_weights_quantization:
                quantized_kernel, io_channels_axes = get_quantized_kernel_by_weights_qc(fw_info,
                                                                                        node,
                                                                                        weights_qc,
                                                                                        fw_impl=fw_impl)

                # If a kernel was quantized and weights bias correction is enabled in n.quantization_cfg,
                # a bias correction term is being calculated and used in the node's bias term.
                if weights_qc.weights_bias_correction:
                    bias_correction_term = _get_bias_correction_term_of_node(io_channels_axes[0],
                                                                             node,
                                                                             node_in_stats_collector,
                                                                             io_channels_axes[1],
                                                                             quantized_kernel,
                                                                             fw_impl=fw_impl)

                    # Store the correction term to use it later,
                    weights_qc.bias_corrected = bias_correction_term


def _compute_bias_correction(kernel: np.ndarray,
                             quantized_kernel: np.ndarray,
                             in_statistics_container: BaseStatsContainer,
                             output_channels_axis: int,
                             input_channels: int) -> Any:
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


def _get_bias_correction_term_of_node(input_channels_axis: int,
                                      n: BaseNode,
                                      node_in_stats_collector: BaseStatsContainer,
                                      output_channels_axis: int,
                                      quantized_kernel: np.ndarray,
                                      fw_impl: FrameworkImplementation):
    """
    Get the bias correction term for a node, using a quantized kernel (which can be quantized
    using any possible bit width)

    Args:
        input_channels_axis: Index of input channels of the kernel.
        n: Node to compute the bias-correction term.
        node_in_stats_collector: Input statistics collector of the node.
        output_channels_axis: Index of output channels of the kernel.
        quantized_kernel: Quantized kernel of the node.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.


    Returns:
        Bias-correction term to subtract from the current node's bias.
    """

    if output_channels_axis is None:
        Logger.error(
            f'Unknown output channel axis for node named: {n.name},'
            f' please update channel mapping function')
    if input_channels_axis is None:
        Logger.error(
            f'Unknown input channel axis for node named: {n.name},'
            f' please update channel mapping function')
    # Compute the bias correction term.
    correction = _compute_bias_correction(n.get_weights_by_keys(fw_impl.constants.KERNEL),
                                          quantized_kernel,
                                          node_in_stats_collector,
                                          output_channels_axis,
                                          input_channels_axis)
    return correction
