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


from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Conv2DTranspose, Dense, ReLU, Add, Activation, PReLU, ELU
from typing import Tuple

from model_compression_toolkit import common
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher, EdgeMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.keras.constants import LINEAR, ACTIVATION


class MarkActivation(common.BaseSubstitution):
    """
    There are cases we do not need to quantize a layer's output since there is an
    activation layer that follows it. Thus, in these cases we set the node's attribute
    that indicates whether the output should be quantized to False.
    """

    def __init__(self):
        """
        Matches: (DepthwiseConv2D,Conv2D,Conv2DTranspose,Dense)[activation=linear] -> (Activation,Relu)
                or
                Add -> (Activation,Relu)
        """
        act_linear = NodeFrameworkAttrMatcher(ACTIVATION, LINEAR)
        source_node = (NodeOperationMatcher(DepthwiseConv2D) |
                       NodeOperationMatcher(Conv2D) |
                       NodeOperationMatcher(Conv2DTranspose) |
                       NodeOperationMatcher(Dense)) & act_linear

        activation_node = NodeOperationMatcher(ReLU) | \
                          NodeOperationMatcher(Activation) | \
                          NodeOperationMatcher(PReLU)

        source_node_add = NodeOperationMatcher(Add)

        super().__init__(
            matcher_instance=EdgeMatcher(source_node, activation_node) | EdgeMatcher(source_node_add, activation_node))

    def substitute(self,
                   graph: Graph,
                   edge: Tuple[BaseNode, BaseNode]) -> Graph:
        """
        Mark the first node in an edge that should not be quantized as so.
        This can be done due to the following reasons:
        1. The second node in the edge is an activation layer.

        Args:
            graph: Graph we apply the substitution on.
            edge: Edge where the first node's output should not be quantized.

        Returns:
            Graph after applying the substitution.
        """

        edge[0].output_quantization = False
        return graph
