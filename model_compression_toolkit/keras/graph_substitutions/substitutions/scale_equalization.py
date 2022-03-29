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


import logging
from typing import List, Dict, Tuple

import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dense, Conv2DTranspose, Activation, ReLU, ZeroPadding2D

from model_compression_toolkit import common
from model_compression_toolkit.common import Graph, BaseNode
from model_compression_toolkit.common.constants import OUTPUT_SCALE, THRESHOLD
from model_compression_toolkit.common.defaultdict import DefaultDict
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher, WalkMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.keras.constants import KERNEL, BIAS, LINEAR, ACTIVATION, RELU_MAX_VALUE
from model_compression_toolkit.keras.constants import RELU


# Match linear layers.
op2d_node = NodeOperationMatcher(DepthwiseConv2D) | \
            NodeOperationMatcher(Conv2D) | \
            NodeOperationMatcher(Conv2DTranspose) | \
            NodeOperationMatcher(Dense)

# Match Conv2D where its activation function keeps f(ax)==af(x)
homogeneous_activation_nodes = op2d_node & (NodeFrameworkAttrMatcher(ACTIVATION, RELU) |
                                            NodeFrameworkAttrMatcher(ACTIVATION, LINEAR))

zeropad_node = NodeOperationMatcher(ZeroPadding2D)

# The substitution is also possible for cases when there's a non-linearity
# between the two linear layers which keeps f(ax)==af(x)
mid_activation_nodes = (NodeOperationMatcher(Activation) &
                        NodeFrameworkAttrMatcher(ACTIVATION, RELU)) | \
                       NodeOperationMatcher(ReLU)

# Two cases to match: linear_op -> linear_op, linear_op -> non_linearity -> linear_op
# Two substitutions do the same thing but match different patterns.
MATCHER = WalkMatcher([homogeneous_activation_nodes, op2d_node])
MATCHER_WITH_PAD = WalkMatcher([homogeneous_activation_nodes, zeropad_node, op2d_node])
MATCHER_MID = WalkMatcher([op2d_node, mid_activation_nodes, op2d_node])
MATCHER_MID_WITH_PAD = WalkMatcher([op2d_node, mid_activation_nodes, zeropad_node, op2d_node])


def scale_reshaping(scale: np.ndarray,
                    op2d: common.BaseNode,
                    kernel_channel_mapping: DefaultDict,
                    in_channels: bool = True) -> np.ndarray:
    """
    Before scaling a kernel, the scale factor needs is reshaped to the correct
    dimensions. This is a function of the layer that is scaled and whether its input channels or
    output channels that should be scaled.
    The index of the correct kernel index is obtained from kernel_channel_mapping.

    Args:
        scale: Scale factor to scale the kernel channels by.
        op2d: Node to scale its kernel.
        kernel_channel_mapping: Mapping from a layer to a tuple of indices of its output/input kernel channels.
        in_channels: Kernel's index of input channels.

    Returns:
        The scale factor after reshaping it to the correct shape.
    """

    op_ndims = op2d.get_weights_by_keys(KERNEL).ndim
    reshape_target = np.ones(op_ndims, dtype=np.int)
    reshape_target[kernel_channel_mapping.get(op2d.type)[int(in_channels)]] = -1
    return np.reshape(scale, reshape_target)


def update_linear_nodes(graph:Graph,
                        qc: QuantizationConfig,
                        fw_info: FrameworkInfo,
                        first_op2d_node: BaseNode,
                        second_op2d_node: BaseNode,
                        scale_factor: np.ndarray):
    """
    Scale the weights of two linear nodes with a scale factor. Each node is scaled in
    the opposite scale factor such that the output of the second node is the same as it
    is without the scaling.
    Thresholds are recalculated as the weights were changed.
    The scale factor contain a scale value per-channel.

    Args:
        graph: Graph to apply the scaling on its nodes.
        qc: QuantizationConfig containing parameters of how the model should be quantized.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        first_op2d_node: Node to multiply its kernel by the scale factor.
        second_op2d_node: Node to divide its kernel by the scale factor.
        scale_factor: Scaling factor to scale the nodes' weights.

    """

    w2_fixed = second_op2d_node.get_weights_by_keys(KERNEL) / scale_reshaping(scale_factor,
                                                                              second_op2d_node,
                                                                              fw_info.kernel_channels_mapping)

    w1_fixed = first_op2d_node.get_weights_by_keys(KERNEL) * scale_reshaping(scale_factor,
                                                                             first_op2d_node,
                                                                             fw_info.kernel_channels_mapping,
                                                                             in_channels=False)

    if first_op2d_node.get_weights_by_keys(BIAS) is not None:
        b1_fixed = first_op2d_node.get_weights_by_keys(BIAS) * scale_factor
        first_op2d_node.set_weights_by_keys(BIAS, b1_fixed)

    first_op2d_node.quantization_attr[OUTPUT_SCALE] = scale_factor
    first_op2d_node.set_weights_by_keys(KERNEL, w1_fixed)
    second_op2d_node.set_weights_by_keys(KERNEL, w2_fixed)

    for nqc in first_op2d_node.candidates_quantization_cfg:
        nqc.weights_quantization_cfg.calculate_and_set_weights_params(w1_fixed)
    for nqc in second_op2d_node.candidates_quantization_cfg:
        nqc.weights_quantization_cfg.calculate_and_set_weights_params(w2_fixed)


def calculate_scale_correction(graph: Graph,
                               activation_node: BaseNode,
                               eps: float = 1e-6) -> tuple:
    """
    Compute a scale factor by the activation node threshold and its outputs statistics in
    order to scale all activations such that their maximal values are the activation node's
    constrained threshold.

    Args:
        graph: Graph where the activation node is in.
        activation_node: Activation node to scale its outputs (thus, its previous layer weights).
        eps: Small number to use the the maximal statistics values are zero.

    Returns:
        Tuple of: scaling factor, activation node constrained threshold, outputs maximal value per-channel.
    """

    tensor_stat = graph.get_out_stats_collector(activation_node)

    if not activation_node.is_all_activation_candidates_equal():
        raise Exception("Scale equalization is not supported for more than one activation quantization configuration "
                        "candidate")

    # all candidates have same activation config, so taking the first candidate for calculations
    threshold = activation_node.candidates_quantization_cfg[0].activation_quantization_cfg.activation_quantization_params.get(THRESHOLD)

    max_vector = np.max(np.stack([tensor_stat.mpcc.max_per_channel, np.abs(tensor_stat.mpcc.min_per_channel)], axis=-1),
                        axis=-1)

    scale_factor = threshold / (max_vector + eps)
    scale_factor[max_vector <= 0] = 1
    activation_node.quantization_attr[OUTPUT_SCALE] = scale_factor
    scale_factor = np.maximum(scale_factor, 1)  # Making sure all scale factor are above 1

    graph.scale_stats_collector(activation_node, scale_factor)

    # scale relu bound so f(ax)==af(x)
    if activation_node.type == ReLU and \
            activation_node.framework_attr.get(RELU_MAX_VALUE) is not None:
        activation_node.framework_attr[RELU_MAX_VALUE] = threshold

    return scale_factor, threshold, tensor_stat.mpcc.state


def scale_equalization_lnl(graph: Graph,
                           qc: QuantizationConfig,
                           fw_info: FrameworkInfo,
                           first_op2d_node: BaseNode,
                           n_node: BaseNode,
                           second_op2d_node: BaseNode):
    """
    Compute a scale factor to scale all activation node's outputs such that
    its maximum per-channel is the constrained threshold of the activation node.
    A correction (opposite computed scale) needs to be applied on the linear node that
    follows the activation node to get the same expected output without the scaling.

    Args:
        graph: Graph to apply the scaling on its nodes.
        n_node: Activation node in the middle of the linear nodes.
        qc: QuantizationConfig containing parameters of how the model should be quantized.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        first_op2d_node: Node to multiply its kernel by the scale factor.
        second_op2d_node: Node to divide its kernel by the scale factor.

    """
    scale_factor, threshold, max_array = calculate_scale_correction(graph, n_node)

    common.Logger.debug(f"{first_op2d_node.name} -> output Max Per Channel:{max_array}")
    common.Logger.debug(f'{first_op2d_node.name} -> Threshold value: {threshold}')
    common.Logger.debug(f'{first_op2d_node.name} -> Scale Factor: {scale_factor}')

    update_linear_nodes(graph,
                        qc,
                        fw_info,
                        first_op2d_node,
                        second_op2d_node,
                        scale_factor)


class BaseScaleEqualization(common.BaseSubstitution):
    """
    Substitution to scale the weights of two linear nodes in order to use the entire
    constrained range when activations are quantized.
    Unlike relu_bound_to_power_of_2, scaling here is per-channel and a non-linear node my or may not
    exist between the linear nodes (set nl_index to 0 if it doesn't)
    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo,
                 matcher_instance,
                 nl_index: int = 0):
        """
        Initialize a ScaleEqualization object.
        Args:
            quant_config: QuantizationConfig containing parameters of how the model should be quantized.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)
            matcher_instance: Per substitution matcher instance of type WalkMatcher
            nl_index: Index of non-linear node in list of matched nodes (index 0 means no non-linear in matcher)
        """

        self.quant_config = quant_config
        self.fw_info = fw_info
        self.nl_index = nl_index
        super().__init__(matcher_instance=matcher_instance)

    def substitute(self,
                   graph: Graph,
                   nodes_list: List[BaseNode]) -> Graph:
        """
        Scale each channel of the weights of two linear nodes,
        in order to use the entire constrained range when activations are quantized.
        If the non-linear layer is bounded, the bound is scaled as well.

        Args:
            graph: Graph to apply the substitution on.
            nodes_list: List of nodes that match the pattern in the substitution init.

        Returns:
            Graph after substitution.

        """
        first_op2d_node = nodes_list[0]
        nl_node = nodes_list[self.nl_index]
        second_op2d_node = nodes_list[-1]
        scale_equalization_lnl(graph,
                               self.quant_config,
                               self.fw_info,
                               first_op2d_node,
                               nl_node,
                               second_op2d_node)
        return graph


class ScaleEqualization(BaseScaleEqualization):
    """
    Substitution extends BaseScaleEqualization to the case of Linear-->Linear
    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo):
        """
        Initialize a ScaleEqualization object.
        Args:
            quant_config: QuantizationConfig containing parameters of how the model should be quantized.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)
        """

        super().__init__(quant_config=quant_config, fw_info=fw_info, matcher_instance=MATCHER)


class ScaleEqualizationWithPad(BaseScaleEqualization):
    """
    Substitution extends BaseScaleEqualization to the case of Linear-->ZeroPadding-->Linear
    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo):
        """
        Initialize a ScaleEqualization object.
        Args:
            quant_config: QuantizationConfig containing parameters of how the model should be quantized.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)
        """

        super().__init__(quant_config=quant_config, fw_info=fw_info, matcher_instance=MATCHER_WITH_PAD)


class ScaleEqualizationMidActivation(BaseScaleEqualization):
    """
    Substitution extends BaseScaleEqualization to the case of Linear-->Non-Lnear-->Linear
    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo):
        """
        Init a ScaleEqualizationMidActivation object.

        Args:
            quant_config: QuantizationConfig containing parameters of how the model should be quantized.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)

        """

        self.quant_config = quant_config
        self.fw_info = fw_info

        super().__init__(quant_config=quant_config, fw_info=fw_info, matcher_instance=MATCHER_MID, nl_index=1)


class ScaleEqualizationMidActivationWithPad(BaseScaleEqualization):
    """
    Substitution extends BaseScaleEqualization to the case of Linear-->Non-linear-->ZeroPadding-->Linear
    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo):
        """
        Init a ScaleEqualizationMidActivation object.

        Args:
            quant_config: QuantizationConfig containing parameters of how the model should be quantized.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)

        """

        self.quant_config = quant_config
        self.fw_info = fw_info

        super().__init__(quant_config=quant_config, fw_info=fw_info, matcher_instance=MATCHER_MID_WITH_PAD, nl_index=1)
