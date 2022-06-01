# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from typing import List

import numpy as np
import scipy

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig


# We assume to have Gaussian distribution before the RelU operation
# Hence, the activations after the RelU operation have Rectified Gaussian distribution
# We need to calculate the "fixed" mean and std of the "new" activations
# For more information about Rectified Gaussian distribution:
# https://en.wikipedia.org/wiki/Rectified_Gaussian_distribution

def fixed_mean_after_relu(mu: np.ndarray,
                          std: np.ndarray):
    """
    Calculate fixed mean after relu

    Args:
        mu: Mean vector of the activations before the RelU operation.
        std: Std vector of the activations before the RelU operation.

    Returns:
        The fixed mean vector of the activations before the RelU operation.
    """
    variance = np.power(std, 2)
    mean_pow_2 = np.power(mu, 2)
    prob_const = 1 / np.sqrt(2 * variance * np.pi)
    second_const = np.sqrt(np.pi / 2) * std * mu
    free_const = variance * np.exp(-1 * mean_pow_2 / (2 * variance))
    erf_const = mu / (np.sqrt(2) * std)
    fixed_mean = prob_const * (second_const * (scipy.special.erf(erf_const) + 1) + free_const)
    return fixed_mean


def fixed_second_moment_after_relu(mu: np.ndarray,
                                   std: np.ndarray):
    """
    Calculate fixed std after relu

    Args:
        mu: Mean vector of the activations before the RelU operation.
        std: Std vector of the activations before the RelU operation.

    Returns:
        The fixed std vector of the activations before the RelU operation.
    """
    variance = np.power(std, 2)
    mean_pow_2 = np.power(mu, 2)
    prob_const = 1 / np.sqrt(2 * variance * np.pi)
    second_const = np.sqrt(np.pi / 2) * std * (mean_pow_2 + variance)
    free_const = mu * variance * np.exp(-1 * mean_pow_2 / (2 * variance))
    erf_const = mu / (np.sqrt(2) * std)
    fixed_non_var = prob_const * (second_const * (scipy.special.erf(erf_const) + 1) + free_const)
    return fixed_non_var


def scale_reshaping(scale: np.ndarray,
                    op2d: common.BaseNode,
                    kernel_channel_mapping: DefaultDict,
                    kernel_str: str,
                    in_channels: bool = True) -> np.ndarray:
    """
    Before scaling a kernel, the scale factor needs to be reshaped to the correct
    dimensions. This is a function of the layer that is scaled and whether its input channels or
    output channels that should be scaled.
    The index of the correct kernel index is obtained from kernel_channel_mapping.

    Args:
        scale: Scale factor to scale the kernel channels by.
        op2d: Node to scale its kernel.
        kernel_channel_mapping: Mapping from a layer to a tuple of indices of its output/input kernel channels.
        kernel_str: The framework specific attribute name of the convolution layer's weight/kernel.
        in_channels: Kernel's index of input channels.

    Returns:
        The scale factor after reshaping it to the correct shape.
    """

    op_ndims = op2d.get_weights_by_keys(kernel_str).ndim
    reshape_target = np.ones(op_ndims, dtype=np.int)
    reshape_target[kernel_channel_mapping.get(op2d.type)[int(in_channels)]] = -1
    return np.reshape(scale, reshape_target)


def update_linear_nodes(fw_info: FrameworkInfo,
                        first_op2d_node: BaseNode,
                        second_op2d_node: BaseNode,
                        scale_factor: np.ndarray,
                        kernel_str: str,
                        bias_str: str):
    """
    Scale the weights of two linear nodes with a scale factor. Each node is scaled in
    the opposite scale factor such that the output of the second node is the same as it
    is without the scaling.
    The scale factor contain a scale value per-channel.

    Args:
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        first_op2d_node: Node to multiply its kernel by the scale factor.
        second_op2d_node: Node to divide its kernel by the scale factor.
        scale_factor: Scaling factor to scale the nodes' weights.
        bias_str: The framework specific attribute name of the convolution layer's bias.
        kernel_str: The framework specific attribute name of the convolution layer's weight/kernel.

    """

    w2_fixed = second_op2d_node.get_weights_by_keys(kernel_str) / scale_reshaping(scale_factor,
                                                                                  second_op2d_node,
                                                                                  fw_info.kernel_channels_mapping,
                                                                                  kernel_str)

    w1_fixed = first_op2d_node.get_weights_by_keys(kernel_str) * scale_reshaping(scale_factor,
                                                                                 first_op2d_node,
                                                                                 fw_info.kernel_channels_mapping,
                                                                                 kernel_str,
                                                                                 in_channels=False)

    if first_op2d_node.get_weights_by_keys(bias_str) is not None:
        b1_fixed = first_op2d_node.get_weights_by_keys(bias_str) * scale_factor
        first_op2d_node.set_weights_by_keys(bias_str, b1_fixed)

    first_op2d_node.set_weights_by_keys(kernel_str, w1_fixed)
    second_op2d_node.set_weights_by_keys(kernel_str, w2_fixed)


def calculate_scale_correction(first_op2d_node: BaseNode) -> tuple:
    """
    Compute a scale factor by the activation node's outputs statistics in order to scale the activations by channel.

    Args:
        first_op2d_node: Node to calculate the scale factor by.

    Returns:
        Scaling factor
    """
    std_vector = np.abs(first_op2d_node.prior_info.std_output)
    mean_vector = first_op2d_node.prior_info.mean_output

    fixed_second_moment_vector = fixed_second_moment_after_relu(mean_vector, std_vector)
    fixed_mean_vector = fixed_mean_after_relu(mean_vector, std_vector)
    fixed_std_vector = np.sqrt(fixed_second_moment_vector - np.power(fixed_mean_vector, 2))

    scale_factor = 1.0 / fixed_std_vector
    scale_factor = np.minimum(scale_factor, 1.0)

    return scale_factor


def scale_equalization_lnl(fw_info: FrameworkInfo,
                           first_op2d_node: BaseNode,
                           second_op2d_node: BaseNode,
                           kernel_str: str,
                           bias_str: str):
    """
    Compute a scale factor by the activation node's outputs statistics in order to scale the activations by channel.
    A correction (opposite computed scale) needs to be applied on the linear node that
    follows the activation node to get the same expected output without the scaling.

    Args:
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        first_op2d_node: Node to multiply its kernel by the scale factor.
        second_op2d_node: Node to divide its kernel by the scale factor.
        bias_str: The framework specific attribute name of the convolution layer's bias.
        kernel_str: The framework specific attribute name of the convolution layer's weight/kernel.

    """
    scale_factor = calculate_scale_correction(first_op2d_node)

    update_linear_nodes(fw_info,
                        first_op2d_node,
                        second_op2d_node,
                        scale_factor,
                        kernel_str,
                        bias_str)


class BaseScaleEqualization(common.BaseSubstitution):
    """
    Substitution to scale the weights of two linear nodes in order to use the entire
    constrained range when activations are quantized.
    Unlike relu_bound_to_power_of_2, scaling here is per-channel.
    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo,
                 matcher_instance,
                 kernel_str: str,
                 bias_str: str):
        """
        Initialize a ScaleEqualization object.
        Args:
            quant_config: QuantizationConfig containing parameters of how the model should be quantized.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)
            matcher_instance: Per substitution matcher instance of type WalkMatcher
        """

        self.quant_config = quant_config
        self.fw_info = fw_info
        self.kernel_str = kernel_str
        self.bias_str = bias_str
        super().__init__(matcher_instance=matcher_instance)

    def substitute(self,
                   graph: Graph,
                   nodes_list: List[BaseNode]) -> Graph:
        """
        Scale each channel of the weights of two linear nodes,

        Args:
            graph: Graph to apply the substitution on.
            nodes_list: List of nodes that match the pattern in the substitution init.

        Returns:
            Graph after substitution.

        """
        first_op2d_node = nodes_list[0]
        act_node = nodes_list[1]
        second_op2d_node = nodes_list[-1]
        if first_op2d_node.prior_info.std_output is not None and act_node.is_activation_quantization_enabled():
            scale_equalization_lnl(self.fw_info,
                                   first_op2d_node,
                                   second_op2d_node,
                                   self.kernel_str,
                                   self.bias_str)
        return graph
