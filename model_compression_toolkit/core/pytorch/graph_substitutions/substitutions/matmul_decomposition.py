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
import operator
from typing import List

import numpy as np

from model_compression_toolkit.core.common.graph.base_graph import OutTensor
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common import BaseNode, Graph, BaseSubstitution
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.constants import *
from model_compression_toolkit.logger import Logger

class MatMulParams:
    """
    A data class to hold all relevant parameter shapes for MatMul decomposition
    """

    def __init__(self,
                 matmul_node: BaseNode):
        """
        Extract params for all nodes from matmul node
        Args:
            matmul_node: original MatMul Node

        Naming convention:
            * First parameter - input
            * Second parameter - other
        """
        self.input_in_shape = tuple(matmul_node.input_shape[0])
        self.other_in_shape = tuple(matmul_node.input_shape[1])
        # Step 1 - reshape
        # (B, D_1, ... , D_N, m, p) --> (B, (D_1*...*D_N), m, p)
        self.input_reshape_out_shape = tuple([
            self.input_in_shape[0],
            int(np.prod(self.input_in_shape[1:-2])),
            self.input_in_shape[-2],
            self.input_in_shape[-1]
        ])
        # (B, D_1, ... , D_N, p, n) --> (B, (D_1*...*D_N), p, n)
        self.other_reshape_out_shape = tuple([
            self.other_in_shape[0],
            int(np.prod(self.other_in_shape[1:-2])),
            self.other_in_shape[-2],
            self.other_in_shape[-1]
        ])
        # Step 2 - unbind
        # (B, (D_1*...*D_N), m, p) --> [(B, m, p)] * (D_1*...*D_N)
        self.input_unbind_single_shape = tuple([
            self.input_in_shape[0],
            self.input_in_shape[-2],
            self.input_in_shape[-1]
        ])
        # (B, (D_1*...*D_N), p, n) --> [(B, p, n)] * (D_1*...*D_N)
        self.input_unbind_out_shape = tuple([list(self.input_unbind_single_shape)] * self.input_reshape_out_shape[1])
        self.other_unbind_single_shape = tuple([
            self.other_in_shape[0],
            self.other_in_shape[-2],
            self.other_in_shape[-1]
        ])
        self.other_unbind_out_shape = tuple([list(self.other_unbind_single_shape)] * self.other_reshape_out_shape[1])
        # Step 3 - matmul for loop
        # [(B, m, p)] * (D_1*...*D_N) X [(B, p, n)] * (D_1*...*D_N) --> [(B, m, n)] * (D_1*...*D_N)
        self.single_matmul_shape = [list(self.input_unbind_single_shape[:-1]) + [self.other_unbind_single_shape[-1]]]
        self.final_matmul_shape = (tuple(self.single_matmul_shape[0]),) * self.other_reshape_out_shape[1]
        # Step 4 - stack and reshape all matmul outputs to original dimensions
        # [(B, m, n)] * (D_1*...*D_N) --> (B, (D_1*...*D_N), m, n)
        self.matmul_stack_shape = tuple([
            self.input_unbind_single_shape[0],
            self.input_reshape_out_shape[1],
            self.input_unbind_single_shape[1],
            self.other_unbind_single_shape[2]
        ])
        # (B, (D_1*...*D_N), m, n) --> (B, D_1, ..., D_N, m, n)
        self.final_output_shape = tuple(
            list(self.input_in_shape[:-1]) + [self.other_in_shape[-1]]
        )


class MatMulDecomposition(BaseSubstitution):
    """
    Removes A MatMul node from the graph if one of its inputs has >3 dimensions
    Replaces it with unbind, matmul and stack nodes

    Naming convention:
            * First parameter - input
            * Second parameter - other
    """

    def __init__(self):
        """
        Matches: torch matmul or matmul operator
        """
        func_node = NodeOperationMatcher(torch.matmul) | NodeOperationMatcher(operator.matmul)
        super().__init__(matcher_instance=func_node)

    def substitute(self,
                   graph: Graph,
                   matmul_node: BaseNode) -> Graph:
        """
        Decompose matmul of matrices with >3 dimensions to multiple matmul and reconstruct graph.
        Args:
            graph: Graph we apply the substitution on.
            matmul_node: MatMul node to substitute
        Returns:
            Graph after applying the substitution.
        """

        # If both matrices are already 3D or less, no need to change the graph
        if len(matmul_node.input_shape[0]) <= 3 and len(matmul_node.input_shape[1]) <= 3:
            return graph
        print(matmul_node.name)
        if not np.array_equal(matmul_node.input_shape[0][:-2], matmul_node.input_shape[1][:-2]):
            Logger.critical(f'Mismatch between input dimensions in node {matmul_node.name}.')  # pragma: no cover

        matmul_params = MatMulParams(matmul_node)

        # Reshape inputs
        # (B, D_1, ... , D_N, m, p) --> (B, (D_1*...*D_N), m, p)
        # (B, D_1, ... , D_N, p, n) --> (B, (D_1*...*D_N), p, n)
        input_reshape_node, other_reshape_node = self._reshape_input(
            graph,
            matmul_node,
            matmul_params
        )

        # Unbind inputs
        # (B, (D_1*...*D_N), m, p) --> [(B, m, p)] * (D_1*...*D_N)
        # (B, (D_1*...*D_N), p, n) --> [(B, p, n)] * (D_1*...*D_N)
        input_unbind_node, other_unbind_node = self._unbind_reshaped(
            graph,
            matmul_node,
            input_reshape_node,
            other_reshape_node,
            matmul_params
        )

        # Matmul each pair
        # [(B, m, p)] * (D_1*...*D_N) X [(B, p, n)] * (D_1*...*D_N) --> [(B, m, n)] * (D_1*...*D_N)
        split_matmul_nodes = []
        for idx in range(matmul_params.input_reshape_out_shape[1]):
            split_matmul_node = self._calc_single_matmul(
                graph,
                matmul_node,
                input_unbind_node,
                other_unbind_node,
                idx,
                matmul_params
            )
            split_matmul_nodes.append(split_matmul_node)

        # Stack and reshape all results
        # [(B, m, n)] * (D_1*...*D_N) --> (B, (D_1*...*D_N), m, n)
        # (B, (D_1*...*D_N), m, n) --> (B, D_1, ..., D_N, m, n)
        matmul_out_node = self._stack_reshape_matmul(
            graph,
            matmul_node,
            split_matmul_nodes,
            matmul_params
        )

        # connect edges to new nodes
        self._connect_to_graph(
            graph,
            matmul_node,
            input_reshape_node,
            other_reshape_node,
            matmul_out_node
        )

        # Finally remove the matmul node
        graph.remove_node(matmul_node, new_graph_outputs=[OutTensor(matmul_out_node, 0)])

        return graph

    @staticmethod
    def _reshape_input(graph: Graph,
                              matmul_node: BaseNode,
                              params: MatMulParams) -> List[BaseNode]:
        """
        This method creates the nodes that reshape the input nodes to be 4D before the split

        Args:
            graph: Graph to apply the substitution on.
            matmul_node: MatMul node.
            params: MatMul shape params.

        Returns:
            Input & Other reshape nodes
        """
        input_reshape_node = FunctionalNode(
            name=f'{matmul_node.name}_input_reshape',
            framework_attr={},
            input_shape=params.input_in_shape,
            output_shape=params.input_reshape_out_shape,
            weights={},
            layer_class=torch.reshape,
            op_call_args=[params.input_reshape_out_shape],
            op_call_kwargs={},
            functional_op=torch.reshape
        )
        other_reshape_node = FunctionalNode(
            name=f'{matmul_node.name}_other_reshape',
            framework_attr={},
            input_shape=params.other_in_shape,
            output_shape=params.other_reshape_out_shape,
            weights={},
            layer_class=torch.reshape,
            op_call_args=[params.other_reshape_out_shape],
            op_call_kwargs={},
            functional_op=torch.reshape
        )
        # Connect reshapes to graph
        # input_in_edge, other_in_edge = graph.in_edges(matmul_node)
        graph.add_node(input_reshape_node)
        # graph.add_edge(input_in_edge[0], input_reshape_node, **graph.get_edge_data(*input_in_edge, 0))
        graph.add_node(other_reshape_node)
        # graph.add_edge(other_in_edge[0], other_reshape_node, **graph.get_edge_data(*other_in_edge, 0))

        return input_reshape_node, other_reshape_node

    @staticmethod
    def _unbind_reshaped(graph: Graph,
                        matmul_node: BaseNode,
                        input_reshape_node: BaseNode,
                        other_reshape_node: BaseNode,
                        params: MatMulParams) -> List[BaseNode]:
        """
        This method creates the nodes that split the parameters from 4D to 3D for single MatMul operation

        Args:
            graph: Graph to apply the substitution on.
            matmul_node: MatMul node.
            input_reshape_node: input node after reshape to 4D
            other_reshape_node: other node after reshape to 4D
            params: MatMul shape params.

        Returns:
            Input & Other unbind nodes - output of each is list of 3D matrices
        """
        input_unbind_node = FunctionalNode(
            name=f'{matmul_node.name}_input_split',
            framework_attr={},
            input_shape=params.input_reshape_out_shape,
            output_shape=params.input_unbind_out_shape,
            weights={},
            layer_class=torch.unbind,
            op_call_args=[1],  # Should this be in kwargs or args
            op_call_kwargs={},
            functional_op=torch.unbind
        )

        other_unbind_node = FunctionalNode(
            name=f'{matmul_node.name}_input_split',
            framework_attr={},
            input_shape=params.other_reshape_out_shape,
            output_shape=params.other_unbind_out_shape,
            weights={},
            layer_class=torch.unbind,
            op_call_args=[1],  # Should this be in kwargs or args
            op_call_kwargs={},
            functional_op=torch.unbind
        )

        # Connect to graph
        graph.add_node_with_in_edges(input_unbind_node, [input_reshape_node])
        graph.add_node_with_in_edges(other_unbind_node, [other_reshape_node])

        return input_unbind_node, other_unbind_node

    @staticmethod
    def _calc_single_matmul(graph: Graph,
                            matmul_node: BaseNode,
                            input_unbind_node: BaseNode,
                            other_unbind_node: BaseNode,
                            dim_index: int,
                            params: MatMulParams) -> BaseNode:
        """
        This method creates the per matmul of each channel

        Args:
            graph: Graph to apply the substitution on.
            matmul_node: Original Matmul node
            input_unbind_node: input after reshape and split
            other_unbind_node: other after reshape and split
            dim_index: index to run matmul on
            params: MatMul Params

        Returns:
            Node after matmul of single dimension
        """
        # (B, m, n) X (B, n, p) -> (B, m, p)
        # Get the input in index dim_index
        get_input_node = FunctionalNode(
            name=f'{matmul_node.name}_input_{dim_index}',
            framework_attr={},
            input_shape=params.input_unbind_out_shape,
            output_shape=params.input_unbind_single_shape,
            weights={},
            layer_class=operator.getitem,
            op_call_args=[dim_index],
            op_call_kwargs={},
            functional_op=operator.getitem
        )
        graph.add_node_with_in_edges(get_input_node, [input_unbind_node], [dim_index])

        # Get the other in index dim_index
        get_other_node = FunctionalNode(
            name=f'{matmul_node.name}_other_{dim_index}',
            framework_attr={},
            input_shape=params.other_unbind_out_shape,
            output_shape=params.other_unbind_single_shape,
            weights={},
            layer_class=operator.getitem,
            op_call_args=[dim_index],
            op_call_kwargs={},
            functional_op=operator.getitem
        )
        graph.add_node_with_in_edges(get_other_node, [other_unbind_node], [dim_index])

        matmul_node = FunctionalNode(name=f'{matmul_node.name}_matmul_{dim_index}',
                                     framework_attr={},
                                     input_shape=(params.input_unbind_single_shape, params.other_unbind_single_shape),
                                     output_shape=params.single_matmul_shape,
                                     weights={},
                                     layer_class=torch.matmul,
                                     op_call_args=[],
                                     op_call_kwargs={},
                                     functional_op=torch.matmul)
        graph.add_node_with_in_edges(matmul_node, [get_input_node, get_other_node])

        return matmul_node

    @staticmethod
    def _stack_reshape_matmul(graph: Graph,
                              matmul_node: BaseNode,
                              split_matmul_nodes: List[BaseNode],
                              params: MatMulParams) -> BaseNode:
        """
        This method creates the nodes for the final stack and reshape for output

        Args:
            graph: Graph to apply the substitution on.
            matmul_node: Original Matmul node
            split_matmul_nodes: list of all single matmul nodes
            params: MatMul Params

        Returns:
            Node after stack and reshape - final output
        """
        # [(B, m, n)] * (D_1*...*D_N) --> (B, (D_1*...*D_N), m, n)
        stack_node = FunctionalNode(
            name=f'{matmul_node.name}_stack',
            framework_attr={DIM: 1},
            input_shape=params.final_matmul_shape,
            output_shape=params.matmul_stack_shape,
            weights={},
            layer_class=torch.stack,
            op_call_args=[],
            op_call_kwargs={DIM: 1},
            functional_op=torch.stack,
            inputs_as_list=True
        )
        graph.add_node_with_in_edges(stack_node, split_matmul_nodes)

        # (B, (D_1 * ... * D_N), m, n) --> (B, D_1, ..., D_N, m, n)
        matmul_reshape_node = FunctionalNode(
            name=f'{matmul_node.name}_reshape',
            framework_attr={},
            input_shape=params.matmul_stack_shape,
            output_shape=params.final_output_shape,
            weights={},
            layer_class=torch.reshape,
            op_call_args=[params.final_output_shape],
            op_call_kwargs={},
            functional_op=torch.reshape
        )
        graph.add_node_with_in_edges(matmul_reshape_node, [stack_node])

        return matmul_reshape_node

    @staticmethod
    def _connect_to_graph(
            graph: Graph,
            matmul_node: BaseNode,
            input_reshape_node: BaseNode,
            other_reshape_node: BaseNode,
            matmul_out_node: BaseNode):
        """
        connect subgraph to input graph
        Args:
            graph: input graph
            matmul_node: MatMul node to subsitute inputs and outputs with
            input_reshape_node: 1st input to MatMul Node
            other_reshape_node: 2nd input to MatMul Node
            matmul_out_node: output node of MatMul
        """
        input_in_edge, other_in_edge = graph.in_edges(matmul_node)
        if graph.get_edge_data(*input_in_edge, 0).get('sink_index') == 0:
            graph.add_edge(input_in_edge[0], input_reshape_node, **graph.get_edge_data(*input_in_edge, 0))
            graph.add_edge(other_in_edge[0], other_reshape_node, **graph.get_edge_data(*other_in_edge, 0))
        else:
            graph.add_edge(input_in_edge[0], other_reshape_node, **graph.get_edge_data(*input_in_edge, 0))
            graph.add_edge(other_in_edge[0], input_reshape_node, **graph.get_edge_data(*other_in_edge, 0))
        graph.remove_edge(input_in_edge[0], matmul_node)
        graph.remove_edge(other_in_edge[0], matmul_node)
        graph.reconnect_out_edges(current_node=matmul_node, new_node=matmul_out_node)
