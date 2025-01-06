# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from torch import nn
import torch.nn.functional as F
import operator
from copy import copy
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common import BaseNode, Graph, BaseSubstitution
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.constants import *
from model_compression_toolkit.logger import Logger


class LinearInputReshape(BaseSubstitution):
    """
    Reshape linear layer input of 3 dimensions to 4 dimensions.
    """

    def __init__(self):
        """
        Matches: functional linear
        """
        linear_node = (
                    NodeOperationMatcher(F.linear) |
                    NodeOperationMatcher(torch.nn.Linear) |
                    NodeOperationMatcher(torch.matmul) |
                    NodeOperationMatcher(operator.matmul)
                     )
        super().__init__(matcher_instance=linear_node)

    def _get_input_reshape_node(self, node):

        output_shape = copy(node.output_shape[0])
        new_dim = 1 if -1 in output_shape else 1 #changed
        output_shape.insert(1, new_dim)
        output_shape = tuple(output_shape)
        reshape_node = FunctionalNode(
            name=f'{node.name}_reshape_in',
            framework_attr={},
            input_shape=node.output_shape,#changed
            output_shape=output_shape,
            weights={},
            layer_class=torch.reshape,
            op_call_args=output_shape,
            op_call_kwargs={},
            functional_op=torch.reshape
        )
        return reshape_node

    def _get_output_reshape_node(self, node):

        output_shape = list(copy(node.output_shape[0]))
        output_shape.pop(1)
        output_shape = [output_shape]
        reshape_node = FunctionalNode(
            name=f'{node.name}_reshape_out',
            framework_attr={},
            input_shape=node.output_shape,
            output_shape=tuple(output_shape[0]),
            weights={},
            layer_class=torch.reshape,
            op_call_args=tuple(output_shape[0]),
            op_call_kwargs={},
            functional_op=torch.reshape
        )
        return reshape_node


    def substitute(self,
                   graph: Graph,
                   linear_node: FunctionalNode) -> Graph:
        """
        Substitute functional.linear and its inputs with Linear.
        Args:
            graph: Graph we apply the substitution on.
            node: node that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """

        # Create new node of layer Linear
        print("linear node:", linear_node.name)

        if len(linear_node.input_shape) != 1 or len(linear_node.input_shape[0]) != 3:
            return graph
        input_node = graph.get_prev_nodes(linear_node)[0]

        reshape_node = self._get_input_reshape_node(input_node)
        graph.add_node_with_in_edges(reshape_node, [input_node])
        linear_node.input_shape = [list(reshape_node.output_shape)]
        linear_node.output_shape[0].insert(1,-1)
        e_attr = graph.get_edge_data(input_node, linear_node)
        graph.add_edge(reshape_node, linear_node, **(e_attr[0]))
        graph.remove_edge(input_node, linear_node)

        reshape_output = self._get_output_reshape_node(linear_node)
        # graph.add_node_with_in_edges(reshape_output, [linear_node])
        graph.reconnect_out_edges(linear_node, reshape_output)
        e_attr = {0: {'source_index': 0, 'sink_index': 0}}
        graph.add_edge(linear_node, reshape_output, **(e_attr[0]))
        return graph
