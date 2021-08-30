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


from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Conv2DTranspose, ReLU, Add, Activation, PReLU, ELU
from typing import Tuple

from sony_model_optimization_package import common
from sony_model_optimization_package.common.graph.base_graph import Graph
from sony_model_optimization_package.common.graph.graph_matchers import NodeOperationMatcher, EdgeMatcher, \
    NodeFrameworkAttrMatcher
from sony_model_optimization_package.common.graph.node import Node
from sony_model_optimization_package.keras.constants import LINEAR, ACTIVATION


class MarkActivation(common.BaseSubstitution):
    """
    There are cases we do not need to quantize a layer's output since there is an
    activation layer that follows it. Thus, in these cases we set the node's attribute
    that indicates whether the output should be quantized to False.
    """

    def __init__(self):
        """
        Matches: (DepthwiseConv2D,Conv2D,Conv2DTranspose)[activation=linear] -> (Activation,Relu)
                or
                Add -> (Activation,Relu)
        """
        act_linear = NodeFrameworkAttrMatcher(ACTIVATION, LINEAR)
        source_node = (NodeOperationMatcher(DepthwiseConv2D) |
                       NodeOperationMatcher(Conv2D) |
                       NodeOperationMatcher(Conv2DTranspose)) & act_linear

        activation_node = NodeOperationMatcher(ReLU) | \
                          NodeOperationMatcher(Activation) | \
                          NodeOperationMatcher(PReLU)

        source_node_add = NodeOperationMatcher(Add)

        super().__init__(
            matcher_instance=EdgeMatcher(source_node, activation_node) | EdgeMatcher(source_node_add, activation_node))

    def substitute(self,
                   graph: Graph,
                   edge: Tuple[Node, Node]) -> Graph:
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
