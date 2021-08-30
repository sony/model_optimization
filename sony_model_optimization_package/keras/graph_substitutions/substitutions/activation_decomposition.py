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


from tensorflow.keras.layers import Dense, DepthwiseConv2D, Conv2D, Conv2DTranspose, Activation, SeparableConv2D

from sony_model_optimization_package import common
from sony_model_optimization_package.common.constants import FLOAT_32, DATA_TYPE
from sony_model_optimization_package.common.graph.base_graph import Graph
from sony_model_optimization_package.common.graph.graph_matchers import NodeOperationMatcher, \
    NodeFrameworkAttrMatcher
from sony_model_optimization_package.common.graph.node import Node
from sony_model_optimization_package.keras.constants import LINEAR, ACTIVATION, TRAINABLE, LAYER_NAME


class ActivationDecomposition(common.BaseSubstitution):
    """
    Replace a linear layer that has an activation function, with two nodes: same linear layer without
    an activation function, and a new activation layer to replace the function the linear node had.
    """

    def __init__(self):
        """
        Matches: (DepthwiseConv2D, Conv2D, Dense, Conv2DTranspose, SeparableConv2D)[activation != identity]
        """
        op2d_node = NodeOperationMatcher(DepthwiseConv2D) | \
                    NodeOperationMatcher(Conv2D) | \
                    NodeOperationMatcher(Dense) | \
                    NodeOperationMatcher(Conv2DTranspose) | \
                    NodeOperationMatcher(SeparableConv2D)

        act_attr = NodeFrameworkAttrMatcher(ACTIVATION, LINEAR).logic_not() & \
                   NodeFrameworkAttrMatcher(ACTIVATION, None).logic_not()

        op2d_node = op2d_node & act_attr
        super().__init__(matcher_instance=op2d_node)

    def substitute(self,
                   graph: Graph,
                   op2d_node: Node) -> Graph:
        """
        Decompose the activation function in a linear node to a new activation layer.
        Set activation function in the linear node to 'linear' (y=x).

        Args:
            graph: Graph we apply the substitution on.
            op2d_node: Node to extract its activation function.

        Returns:
            Graph after applying the substitution.
        """

        activation_node_name = op2d_node.name + '_post_activation'

        activation_fw_attr = {
            LAYER_NAME: activation_node_name,
            TRAINABLE: False,
            DATA_TYPE: FLOAT_32,
            ACTIVATION: op2d_node.framework_attr.get(ACTIVATION)}

        activation_node = common.graph.Node(activation_node_name,
                                            activation_fw_attr,
                                            op2d_node.output_shape,
                                            op2d_node.output_shape,
                                            {},
                                            Activation)

        graph.add_node(activation_node)
        graph.reconnect_out_edges(current_node=op2d_node,
                                  new_node=activation_node)
        graph.add_edge(op2d_node,
                       activation_node,
                       source_index=0,
                       sink_index=0)

        op2d_node.framework_attr[ACTIVATION] = LINEAR
        graph.replace_output_node(current_node=op2d_node,
                                  new_node=activation_node)

        return graph
