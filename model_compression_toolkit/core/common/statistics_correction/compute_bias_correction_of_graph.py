# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

import numpy as np

from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.quantization.quantize_node import get_quantized_weights_attr_by_qc
from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector
from model_compression_toolkit.logger import Logger


def compute_bias_correction_of_graph(graph: Graph,
                                     fw_info: FrameworkInfo,
                                     fw_impl: FrameworkImplementation) -> Graph:
    """
    For each node in a graph, and for each candidate weights quantization configuration,
    compute the bias-correction term, and store it in the candidate weights quantization configuration.

    Args:
        graph: Graph with nodes to compute the bias correction for
        each node's weights quantization configuration candidates.
        fw_info: Framework info like lists of nodes their kernel should quantized.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Graph with bias correction for each weights quantization configuration candidate
        for each node.
    """

    for n in graph.nodes:
        # Bias correction is computed based on the quantized kernel, so we need to get the specific kernel attribute
        # name out of all the weights attributes of the node.
        if fw_info.is_kernel_op(n.type):
            kernel_attr = fw_info.get_kernel_op_attributes(n.type)[0]
            if n.is_weights_quantization_enabled(kernel_attr):
                # Bias correction is not applied to layers with constant inputs.
                if n.has_positional_weights:
                    for candidate_qc in n.candidates_quantization_cfg:
                        candidate_qc.weights_quantization_cfg.weights_bias_correction = False
                else:
                    _compute_bias_correction_per_candidate_qc(n,
                                                              kernel_attr,
                                                              fw_info,
                                                              graph.get_in_stats_collector(n),
                                                              fw_impl=fw_impl)
    return graph


def _compute_bias_correction_per_candidate_qc(node: BaseNode,
                                              kernel_attr: str,
                                              fw_info: FrameworkInfo,
                                              node_in_stats_collector: BaseStatsCollector,
                                              fw_impl: FrameworkImplementation):
    """
    For each candidate weights quantization configuration of a given node,
    compute the bias-correction term, and store it in the candidate weights quantization configuration.

    Args:
        node: Node to compute the bias correction for its different candidates.
        kernel_attr: The name of the kernel attribute of the node.
        fw_info: Framework info like lists of nodes their kernel should quantized.
        node_in_stats_collector: Statistics collector of the node for the mean per-channel.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    """

    for candidate_qc in node.candidates_quantization_cfg:
        if candidate_qc.weights_quantization_cfg.weights_bias_correction and not \
                candidate_qc.weights_quantization_cfg.weights_second_moment_correction:

            quantized_kernel, io_channels_axes = get_quantized_weights_attr_by_qc(kernel_attr,
                                                                                  node,
                                                                                  candidate_qc.weights_quantization_cfg
                                                                                  .get_attr_config(kernel_attr))

            bias_correction_term = _get_bias_correction_term_of_node(io_channels_axes[0],
                                                                     node,
                                                                     node_in_stats_collector,
                                                                     io_channels_axes[1],
                                                                     quantized_kernel,
                                                                     fw_impl=fw_impl)

            # Store the correction term to use it later,
            candidate_qc.weights_quantization_cfg.bias_corrected = bias_correction_term


def is_non_positive_integer(x: float) -> bool:
    """
    Check if a variable is positive integer or not
    Args:
        x: input float to check
    Returns:
        True if x is non-positive integer
    """
    return x < 1 or int(x) != x


def _compute_bias_correction(kernel: np.ndarray,
                             quantized_kernel: np.ndarray,
                             in_statistics_container: BaseStatsCollector,
                             output_channels_axis: int,
                             input_channels_axis: int) -> Any:
    """
    Compute the bias correction term for the bias in the error on the layer’s output,
    that is introduced by the weights quantization.
    For more info: https://arxiv.org/abs/1906.04721

    Args:
        kernel: Float kernel of the layer that its output is biased.
        quantized_kernel: Quantized kernel of the layer that its output is biased.
        in_statistics_container: Inputs statistics of the quantized layer that has the bias error.
        output_channels_axis: Output channels index of the given kernel.
        input_channels_axis: Input channels index of the given kernel.

    Returns:
        Term to add to the bias of the quantized layer in order to correct the expected
        bias due to weights quantization.
    """

    quantization_error = quantized_kernel - kernel
    mu = in_statistics_container.get_mean()
    axis_not_input_output_channel = tuple(
        [i for i in range(len(quantization_error.shape)) if i not in [output_channels_axis, input_channels_axis]])
    eps = np.sum(quantization_error, axis=axis_not_input_output_channel)

    # A special case for Tenesorflow DepthwiseConv2D
    if output_channels_axis == input_channels_axis:
        # Tensorflow's kerenl dimensions: [h, w, in_channels, depth_multiplier]
        eps = np.sum(quantization_error, axis=(0, 1))  # Sum noises over h,w
        eps = eps.reshape((-1, 1)) # Prepare shape: (num_output_channels, depth_of_each_kernel)

    if output_channels_axis > input_channels_axis:
        eps = np.transpose(eps)

    num_groups = mu.shape[0] / eps.shape[1]
    num_out_channels = eps.shape[0] # 0 is always the output channel axis in eps
    correction_term = np.zeros(num_out_channels)

    # Sanity validation
    if is_non_positive_integer(num_groups) or is_non_positive_integer(num_out_channels / num_groups):
        Logger.warning("Skipping bias correction due to valiation problem.")
        return correction_term

    num_out_channels_per_group = int(num_out_channels / num_groups)

    # In Pytorch the output of group conv is separated into respective groups is
    # viewed as follows: (batch, channel, ngroups, h, w),
    # i.e each group is consistently viewed one after the other
    # For an example, check out: https://discuss.pytorch.org/t/group-convolution-output-order/88258
    mu_split = np.split(mu, num_groups)
    eps_split = np.split(eps, num_groups, 0)
    for i, (mu_s, eps_s) in enumerate(zip(mu_split, eps_split)):
        correction_term[i * num_out_channels_per_group:(i + 1) * num_out_channels_per_group] = np.matmul(eps_s, mu_s)

    return correction_term


def _get_bias_correction_term_of_node(input_channels_axis: int,
                                      n: BaseNode,
                                      node_in_stats_collector: BaseStatsCollector,
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
        Logger.critical(
            f'Unknown output channel axis for node: {n.name}. Please update the channel mapping function.')
    if input_channels_axis is None:
        Logger.critical(
            f'Unknown input channel axis for node: {n.name}. Please update the channel mapping function')
    # Compute the bias correction term.
    correction = _compute_bias_correction(n.get_weights_by_keys(fw_impl.constants.KERNEL),
                                          quantized_kernel,
                                          node_in_stats_collector,
                                          output_channels_axis,
                                          input_channels_axis)
    return correction
