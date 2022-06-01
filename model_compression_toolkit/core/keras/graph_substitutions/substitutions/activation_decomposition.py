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


from tensorflow.keras.layers import Dense, DepthwiseConv2D, Conv2D, Conv2DTranspose, Activation, SeparableConv2D

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.constants import FLOAT_32, DATA_TYPE
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.keras.constants import LINEAR, ACTIVATION, TRAINABLE, LAYER_NAME


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
                   op2d_node: BaseNode) -> Graph:
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

        activation_node = common.graph.BaseNode(activation_node_name,
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
