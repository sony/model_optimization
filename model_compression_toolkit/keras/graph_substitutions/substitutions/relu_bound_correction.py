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


from typing import List

import numpy as np
from tensorflow.keras.layers import ReLU, Activation, DepthwiseConv2D, Conv2DTranspose, Conv2D, Dense

from model_compression_toolkit import common
from model_compression_toolkit.common import FrameworkInfo, Graph, BaseNode
from model_compression_toolkit.common.constants import THRESHOLD
from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher, WalkMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.statistics_collector import is_number
from model_compression_toolkit.keras.constants import KERNEL, BIAS, ACTIVATION, RELU_MAX_VALUE
from model_compression_toolkit.keras.constants import RELU





class ReLUBoundCorrection(common.BaseSubstitution):
    """
    Substitution to scale the weights of two linear nodes, and the bound of non-linear between them
    (if bounded) in order to use the entire constrained range when activations are quantized.
    """

    def __init__(self):
        """
        Initialize a ReLUBoundCorrection object.
        """

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
                   nodes_list: List[BaseNode]) -> Graph:
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
            for nqc in first_op2d_node.candidates_weights_quantization_cfg:
                nqc.calculate_and_set_weights_params(w1_fixed)
            for nqc in second_op2d_node.candidates_weights_quantization_cfg:
                nqc.calculate_and_set_weights_params(w2_fixed)

        return graph