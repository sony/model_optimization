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

import copy
from typing import Tuple, Callable

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import EdgeMatcher, NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode


class ResidualCollapsing(common.BaseSubstitution):
    """
    Collapse Add residual into previous Conv2D (No non-linear activation between them)
    """
    def __init__(self,
                 first_node: NodeOperationMatcher,
                 second_node: NodeOperationMatcher,
                 residual_collapsing_fn: Callable,
                 kernel_str: str,
                 layer_name_str: str = None):
        """
        Collapsing Add residual node into previous Conv2D node
        Args:
            first_node: Node matcher for convolution type nodes.
            second_node: Node matcher for add type nodes.
            residual_collapsing_fn: Function for updating the convolution kernel
            kernel_str: The framework specific attribute name of the convolution layer's weight/kernel.
            layer_name_str: The framework specific attribute name of layer's name.
        """
        super().__init__(matcher_instance=EdgeMatcher(first_node, second_node))
        self.residual_collapsing_fn = residual_collapsing_fn
        self.kernel_str = kernel_str
        self.layer_name_str = layer_name_str

    def substitute(self,
                   graph: Graph,
                   edge_nodes: Tuple[BaseNode, BaseNode]) -> Graph:
        """
        Collapse residual Add layer into previous Conv2D layers.
        Args:
            graph: Graph we apply the substitution on.
            edge_nodes: Tuple of two linear nodes
        Returns:
            Graph after applying the substitution.
        """

        first_node = edge_nodes[0]
        second_node = edge_nodes[1]

        # If the linear operator is part of a reused group (it is the "base" node, or a reused node),
        # we should skip the substitution.
        if first_node.reuse or first_node.reuse_group is not None:
            return graph
        if second_node.reuse or second_node.reuse_group is not None:
            return graph

        # Check if convolution and residual satisfy the collapsing conditions, otherwise skip substitution
        if len(graph.get_next_nodes(first_node)) > 1 or len(graph.get_prev_nodes(second_node)) != 2:
            return graph

        # Check if Add is residual connection, otherwise skip substitution
        conv_prev_nodes = graph.get_prev_nodes(first_node)
        add_prev_nodes = graph.get_prev_nodes(second_node)
        add_prev_nodes.remove(first_node)
        if conv_prev_nodes[0] != add_prev_nodes[0]:
            return graph

        # New collapsed weights
        kernel_collapsed = self.residual_collapsing_fn(first_node, self.kernel_str)

        num_nodes_before_substition = len(graph.nodes)
        num_edges_before_substition = len(graph.edges)

        # New collapsed node
        conv_collapsed = copy.deepcopy(first_node)
        conv_collapsed_name = first_node.name + '_' + second_node.name + "_collapsed"
        conv_collapsed.name = conv_collapsed_name
        conv_collapsed.set_weights_by_keys(self.kernel_str, kernel_collapsed)
        if self.layer_name_str is not None:
            conv_collapsed.framework_attr[self.layer_name_str] = conv_collapsed_name

        # Update graph
        graph.add_node(conv_collapsed)
        graph.reconnect_out_edges(current_node=second_node, new_node=conv_collapsed)
        graph.reconnect_in_edges(current_node=first_node, new_node=conv_collapsed)
        graph.replace_output_node(current_node=second_node, new_node=conv_collapsed)

        graph.remove_edge(first_node, second_node)
        graph.remove_edge(add_prev_nodes[0], second_node)
        graph.remove_node(first_node)
        graph.remove_node(second_node)

        # Sanity check
        assert num_nodes_before_substition - len(graph.nodes) == 1
        assert num_edges_before_substition - len(graph.edges) == 2

        return graph
