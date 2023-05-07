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
from torch import reshape
import torch

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.pytorch.constants import BATCH_DIM_VALUE


class ReshapeWithStaticShapes(common.BaseSubstitution):
    """
    Replace "reshape" or "view" shape attributes. Shape attributes are replaced to static const values.
    """

    def __init__(self):
        """
        Matches: 'reshape' or 'view' operators.
        """
        nodes = NodeOperationMatcher(reshape) | NodeOperationMatcher(torch.Tensor.view)
        super().__init__(matcher_instance=nodes)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        Replaces the 'size' attribute to 'reshape' or 'view' operators to be a list of integers,
        determined by their intended output shape. This replaces 'size' attributes that come from
        nodes in the graph. We delete nodes for which that was their sole purpose.

        Args:
            graph: Graph we apply the substitution on.
            node: node that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """
        # we want the batch size value to infer from the length of the array and remaining dimensions
        if len(node.output_shape) == 1:
            node.output_shape[0][0] = BATCH_DIM_VALUE
        else:
            Logger.error('Reshape or view nodes should have a single output shape')  # pragma: no cover

        # configure the new static output shape attribute
        node.op_call_args = node.output_shape

        # modify the node input info
        node.input_shape = [node.input_shape[0]]

        # the first input is the tensor to be reshaped, we want his batch size value to infer
        # from the length of the array and remaining dimensions
        node.input_shape[0][0] = BATCH_DIM_VALUE

        nodes_to_check = []
        for in_edge in graph.incoming_edges(node):
            if in_edge.sink_index > 0:  # the first input is the tensor to be reshaped
                nodes_to_check.append(in_edge.source_node)
                graph.remove_edge(in_edge.source_node, node)
        for n in nodes_to_check:
            clean_graph_from_nodes_without_out_edges(graph, n)
        return graph


def clean_graph_from_nodes_without_out_edges(graph: Graph,
                                             node: BaseNode):
    """
    This function removes all the nodes and edges to nodes that are not connected to any other node,
    and are not output nodes.
    Args:
        graph: Graph we apply the substitution on.
        node: node that match the pattern in the substitution init.

    """
    output_nodes = [o.node for o in graph.get_outputs()]
    if len(graph.out_edges(node)) == 0 and node not in output_nodes:
        nodes_to_check = []
        for in_edge in graph.incoming_edges(node):
            nodes_to_check.append(in_edge.source_node)
            graph.remove_edge(in_edge.source_node, node)
        graph.remove_node(node)
        for n in nodes_to_check:
            clean_graph_from_nodes_without_out_edges(graph, n)
