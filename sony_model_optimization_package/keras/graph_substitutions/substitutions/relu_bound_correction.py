# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


from typing import List

import numpy as np
from tensorflow.keras.layers import ReLU, Activation, DepthwiseConv2D, Conv2DTranspose, Conv2D, Dense

from sony_model_optimization_package import common
from sony_model_optimization_package.common import FrameworkInfo, Graph, Node
from sony_model_optimization_package.common.constants import THRESHOLD
from sony_model_optimization_package.common.graph.graph_matchers import NodeOperationMatcher, WalkMatcher, \
    NodeFrameworkAttrMatcher
from sony_model_optimization_package.common.quantization.quantization_config import QuantizationConfig
from sony_model_optimization_package.common.statistics_collector import is_number
from sony_model_optimization_package.keras.constants import KERNEL, BIAS, ACTIVATION, RELU_MAX_VALUE
from sony_model_optimization_package.keras.constants import RELU


class ReLUBoundCorrection(common.BaseSubstitution):
    """
    Substitution to scale the weights of two linear nodes, and the bound of non-linear between them
    (if bounded) in order to use the entire constrained range when activations are quantized.
    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo):
        """
        Initialize a ReLUBoundCorrection object.

        Args:
            quant_config: QuantizationConfig containing parameters of how the model should be quantized.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)
        """

        self.fw_info = fw_info
        self.quant_config = quant_config

        homogeneous_activation_nodes = NodeOperationMatcher(ReLU) | \
                                       NodeOperationMatcher(Activation) & \
                                       NodeFrameworkAttrMatcher(ACTIVATION, RELU)

        op2d_node = NodeOperationMatcher(DepthwiseConv2D) | \
                    NodeOperationMatcher(Conv2D) | \
                    NodeOperationMatcher(Conv2DTranspose) | \
                    NodeOperationMatcher(Dense)

        wm = WalkMatcher([op2d_node,
                          homogeneous_activation_nodes,
                          op2d_node])

        super().__init__(matcher_instance=wm)

    def substitute(self,
                   graph: Graph,
                   nodes_list: List[Node]) -> Graph:
        """
        Transform a list of nodes in a graph to use the entire constrained quantized range.
        This is done by scaling two linear nodes with a non-linearity between them, if the non-linearity
        keeps: f(ax) = af(x).
        If the non-linearity is bounded (as ReLU6), the bound needs to be scaled as well.

        Args:
            graph: Graph to apply the substitution on.
            nodes_list: List of nodes that match the pattern in the substitution init.

        Returns:
            Graph after substitution.
        """

        first_op2d_node = nodes_list[0]
        non_linear_node = nodes_list[1]
        second_op2d_node = nodes_list[2]

        max_value = non_linear_node.framework_attr.get(RELU_MAX_VALUE, None)
        if max_value is None:
            return graph

        threshold = non_linear_node.activation_quantization_cfg.activation_quantization_params.get(THRESHOLD)
        threshold_float = max_value

        if threshold > threshold_float:
            scale_factor = threshold_float / threshold

            # Scale activation bound only if it is already bounded and has 'max_value' attribute.
            if RELU_MAX_VALUE in non_linear_node.framework_attr and \
                    is_number(non_linear_node.framework_attr[RELU_MAX_VALUE]):
                non_linear_node.framework_attr[RELU_MAX_VALUE] = np.float32(threshold)
                common.Logger.debug(
                    f"Node named:{non_linear_node.name} max value change "
                    f"to:{non_linear_node.framework_attr[RELU_MAX_VALUE]}")

            w2_fixed = scale_factor * second_op2d_node.get_weights_by_keys(KERNEL)
            w1_fixed = first_op2d_node.get_weights_by_keys(KERNEL) / scale_factor
            b1_fixed = first_op2d_node.get_weights_by_keys(BIAS) / scale_factor

            first_op2d_node.set_weights_by_keys(KERNEL, w1_fixed)
            first_op2d_node.set_weights_by_keys(BIAS, b1_fixed)
            second_op2d_node.set_weights_by_keys(KERNEL, w2_fixed)

            graph.scale_stats_collector(non_linear_node, 1 / scale_factor)
            graph.scale_stats_collector(first_op2d_node, 1 / scale_factor)

            # After scaling weights may have different thresholds so it needs to be recalculated
            first_op2d_node.weights_quantization_cfg.calculate_and_set_weights_params(w1_fixed)
            second_op2d_node.weights_quantization_cfg.calculate_and_set_weights_params(w2_fixed)

        return graph
