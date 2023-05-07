# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

from torch.nn import Conv2d, ReLU6, Linear, ConvTranspose2d, Hardtanh
from torch.nn.functional import relu6, hardtanh

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, WalkMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.pytorch.constants import KERNEL, BIAS, INPLACE, HARDTANH_MIN_VAL, HARDTANH_MAX_VAL, \
    RELU_POT_BOUND
from model_compression_toolkit.logger import Logger


class ReLUBoundToPowerOfTwo(common.BaseSubstitution):
    """
    Substitution to scale the weights of two linear nodes, and move the bound of non-linear between them
    (if bounded) in order to use the entire constrained range when activations are quantized.
    """

    def __init__(self):
        """
        Initialize a ReLUBoundToPowerOfTwo object.
        """
        homogeneous_activation_nodes = NodeOperationMatcher(ReLU6) | \
                                       NodeOperationMatcher(relu6) | \
                                       NodeOperationMatcher(Hardtanh) | \
                                       NodeOperationMatcher(hardtanh)

        op2d_node = NodeOperationMatcher(Conv2d) | \
                    NodeOperationMatcher(Linear) | \
                    NodeOperationMatcher(ConvTranspose2d)

        wm = WalkMatcher([op2d_node,
                          homogeneous_activation_nodes,
                          op2d_node])

        self.threshold = RELU_POT_BOUND
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

        # only act on bound relu with not POT max value and 0 min value
        if non_linear_node.type == ReLU6:
            scale_factor = 6.0 / self.threshold
            non_linear_node.layer_class = Hardtanh
            non_linear_node.framework_attr[INPLACE] = False
            non_linear_node.framework_attr[HARDTANH_MIN_VAL] = 0.0
            non_linear_node.framework_attr[HARDTANH_MAX_VAL] = self.threshold
        elif non_linear_node.type == relu6:
            scale_factor = 6.0 / self.threshold
            non_linear_node.functional_op = hardtanh
            non_linear_node.functional_op.__defaults__ = (0.0, self.threshold, False)
        elif non_linear_node.type == Hardtanh:
            if (non_linear_node.framework_attr[HARDTANH_MIN_VAL] == 0.0) and not \
                    (np.log2(non_linear_node.framework_attr[HARDTANH_MAX_VAL]).astype(int) -
                     np.log2(non_linear_node.framework_attr[HARDTANH_MAX_VAL]) == 0):
                scale_factor = non_linear_node.framework_attr[HARDTANH_MAX_VAL] / self.threshold
                non_linear_node.framework_attr[HARDTANH_MAX_VAL] = self.threshold
            else:
                return graph
        elif non_linear_node.type == hardtanh:
            if (non_linear_node.framework_attr[HARDTANH_MIN_VAL] == 0.0) and not \
                    (np.log2(non_linear_node.framework_attr[HARDTANH_MAX_VAL]).astype(int) -
                     np.log2(non_linear_node.framework_attr[HARDTANH_MAX_VAL]) == 0):
                scale_factor = non_linear_node.framework_attr[HARDTANH_MAX_VAL] / self.threshold
                non_linear_node.functional_op.__defaults__ = (0.0, self.threshold, non_linear_node.framework_attr[INPLACE])
            else:
                return graph
        else:
            Logger.error(f"In substitution with wrong matched pattern")
        Logger.debug(
            f"Node named:{non_linear_node.name} changed "
            f"to:{non_linear_node.type}")

        w2_fixed = scale_factor * second_op2d_node.get_weights_by_keys(KERNEL)
        w1_fixed = first_op2d_node.get_weights_by_keys(KERNEL) / scale_factor
        b1_fixed = first_op2d_node.get_weights_by_keys(BIAS) / scale_factor

        first_op2d_node.set_weights_by_keys(KERNEL, w1_fixed)
        first_op2d_node.set_weights_by_keys(BIAS, b1_fixed)
        second_op2d_node.set_weights_by_keys(KERNEL, w2_fixed)
        return graph
