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
import torch

from model_compression_toolkit.core.common.graph.base_graph import OutTensor
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common import BaseNode, Graph, BaseSubstitution
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.constants import *
from model_compression_toolkit.logger import Logger


class MatMulParams:
    """
    A data class to hold all relevant parameter shapes and nodes for MatMul decomposition.
    """

    def __init__(self,
                 matmul_node: FunctionalNode):
        """
        Extract params for all the substitution nodes from original matmul node.
        Args:
            matmul_node: original MatMul Node

        Naming convention:
            * First parameter - input
            * Second parameter - other
        """
        self.head_input_node, self.head_other_node = None, None
        self.prev_input_node, self.prev_other_node = None, None

        self.input_shape, self.other_shape = matmul_node.input_shape

        # Step 1 - Expand
        expand_shape = np.max(np.vstack((self.input_shape[1:-2], self.other_shape[1:-2])), axis=0).tolist()
        self.input_expand_shape = tuple([-1] + expand_shape + list(self.input_shape[-2:]))
        self.other_expand_shape = tuple([-1] + expand_shape + list(self.other_shape[-2:]))

        # Step 2 - Reshape
        # (B, D_1, ... , D_N, m, p) --> (B, (D_1*...*D_N), m, p)
        self.input_reshape_shape = [
            -1,
            int(np.prod(self.input_expand_shape[1:-2])),
            self.input_expand_shape[-2],
            self.input_expand_shape[-1]
        ]
        # (B, D_1, ... , D_N, p, n) --> (B, (D_1*...*D_N), p, n)
        self.other_reshape_shape = [
            -1,
            int(np.prod(self.other_expand_shape[1:-2])),
            self.other_expand_shape[-2],
            self.other_expand_shape[-1]
        ]

        # Step 3 - Split
        # (B, (D_1*...*D_N), m, p) --> [(B, m, p)] * (D_1*...*D_N)
        self.input_matmul_shape = [-1] + self.input_reshape_shape[-2:]
        self.input_split_shape = tuple([self.input_matmul_shape] * self.input_reshape_shape[1])
        # (B, (D_1*...*D_N), p, n) --> [(B, p, n)] * (D_1*...*D_N)
        self.other_matmul_shape = [-1] + self.other_reshape_shape[-2:]
        self.other_split_shape = tuple([self.other_matmul_shape] * self.other_reshape_shape[1])

        # Step 4 - Matmul loop
        # [(B, m, p)] * (D_1*...*D_N) X [(B, p, n)] * (D_1*...*D_N) --> [(B, m, n)] * (D_1*...*D_N)
        self.single_matmul_shape = self.input_matmul_shape[:-1] + [self.other_matmul_shape[-1]]

        # Step 5 - Stack and Reshape all matmul outputs to original dimensions
        # [(B, m, n)] * (D_1*...*D_N) --> (B, (D_1*...*D_N), m, n)
        self.matmul_stack_shape = tuple([-1] + [self.input_reshape_shape[1]] + self.single_matmul_shape[1:])
        # (B, (D_1*...*D_N), m, n) --> (B, D_1, ..., D_N, m, n)
        self.output_shape = tuple(list(self.input_expand_shape)[:-1] + [self.matmul_stack_shape[-1]])

    def update_nodes(self,
                     input_node: FunctionalNode,
                     other_node: FunctionalNode):
        """
        Updates the head and prev nodes to support the option of skipping unnecessary operations.
        Args:
            input_node: node that operates on the input branch
            other_node: node that operates on the other branch
        """
        if not self.head_input_node:
            self.head_input_node = input_node
        if not self.head_other_node:
            self.head_other_node = other_node
        self.prev_input_node = input_node
        self.prev_other_node = other_node


class MatMulDecomposition(BaseSubstitution):
    """
    Removes A MatMul node from the graph if one of its inputs has >3 dimensions and replaces it with unbind, matmul
    and stack nodes. Substitution is done inplace.

    Naming convention:
            * First parameter - input
            * Second parameter - other
    """

    def __init__(self):
        """
        Matches: torch matmul or matmul operator.
        """
        func_node = NodeOperationMatcher(torch.matmul) | NodeOperationMatcher(operator.matmul)
        super().__init__(matcher_instance=func_node)

    def substitute(self,
                   graph: Graph,
                   matmul_node: FunctionalNode) -> Graph:
        """
        Decompose matmul of matrices with >3 dimensions to multiple matmuls and reconstruct graph.
        Args:
            graph: Graph we apply the substitution on.
            matmul_node: MatMul node to substitute
        Returns:
            A graph after applying the substitution.
        """

        # If both matrices are already 3D or less, no need to change the graph
        if len(matmul_node.input_shape[0]) <= 3 and len(matmul_node.input_shape[1]) <= 3:
            return graph

        if len(matmul_node.input_shape[0]) != len(matmul_node.input_shape[1]):
            Logger.critical(f'Mismatch between number of input dimensions in node {matmul_node.name}.')

        matmul_params = MatMulParams(matmul_node)

        # Expand inputs to equal dimensions (instead of broadcasting) - if needed
        if not np.array_equal(matmul_params.input_shape[1:-2], matmul_params.other_shape[1:-2]):
            input_expand_node, other_expand_node = self._expand_inputs(
                graph,
                matmul_node,
                matmul_params
            )
            matmul_params.update_nodes(input_node=input_expand_node, other_node=other_expand_node)

        # Reshape inputs - if needed
        # (B, D_1, ... , D_N, m, p) --> (B, (D_1*...*D_N), m, p)
        # (B, D_1, ... , D_N, p, n) --> (B, (D_1*...*D_N), p, n)
        if len(matmul_params.input_shape) > 4:  # both input & other have the same number of dimensions
            input_reshape_node, other_reshape_node = self._reshape_input(
                graph,
                matmul_node,
                matmul_params
            )
            matmul_params.update_nodes(input_node=input_reshape_node, other_node=other_reshape_node)

        # Split inputs
        # (B, (D_1*...*D_N), m, p) --> [(B, m, p)] * (D_1*...*D_N)
        # (B, (D_1*...*D_N), p, n) --> [(B, p, n)] * (D_1*...*D_N)
        input_split_node, other_split_node = self._split_inputs(
            graph,
            matmul_node,
            matmul_params
        )
        matmul_params.update_nodes(input_node=input_split_node, other_node=other_split_node)

        # Matmul each pair
        # [(B, m, p)] * (D_1*...*D_N) X [(B, p, n)] * (D_1*...*D_N) --> [(B, m, n)] * (D_1*...*D_N)
        split_matmul_nodes = []
        for idx in range(matmul_params.input_reshape_shape[1]):
            split_matmul_node = self._calc_single_matmul(
                graph,
                matmul_node,
                input_split_node,
                other_split_node,
                idx,
                matmul_params
            )
            split_matmul_nodes.append(split_matmul_node)

        # Stack and reshape all results - reshape if needed
        # [(B, m, n)] * (D_1*...*D_N) --> (B, (D_1*...*D_N), m, n)
        # (B, (D_1*...*D_N), m, n) --> (B, D_1, ..., D_N, m, n)
        output_node = self._stack_matmul_outputs(
            graph,
            matmul_node,
            split_matmul_nodes,
            matmul_params
        )

        # connect edges to new nodes
        self._connect_to_graph(
            graph,
            matmul_node,
            matmul_params.head_input_node,
            matmul_params.head_other_node,
            output_node
        )

        # remove the original matmul node
        graph.remove_node(matmul_node, new_graph_outputs=[OutTensor(output_node, 0)])

        return graph

    @staticmethod
    def _expand_inputs(graph: Graph,
                       matmul_node: FunctionalNode,
                       params: MatMulParams) -> List[FunctionalNode]:
        """
        This method creates the nodes that expand the inputs such that the dimensions fit the MatMul process.

        Args:
            graph: Graph to apply the substitution on.
            matmul_node: MatMul node.
            params: MatMul shape params.

        Returns:
            Input & Other expand nodes.
        """
        if params.input_shape[1:] != list(params.input_expand_shape[1:]):
            input_expand_node = FunctionalNode(
                name=f'{matmul_node.name}_input_expand',
                framework_attr={},
                input_shape=params.input_shape,
                output_shape=params.input_expand_shape,
                weights={},
                layer_class=torch.broadcast_to,
                op_call_args=[params.input_expand_shape],
                op_call_kwargs={},
                functional_op=torch.broadcast_to
            )
            graph.add_node(input_expand_node)
        else:
            input_expand_node = None

        if params.other_shape[1:] != list(params.other_expand_shape[1:]):
            other_expand_node = FunctionalNode(
                name=f'{matmul_node.name}_other_expand',
                framework_attr={},
                input_shape=params.other_shape,
                output_shape=params.other_expand_shape,
                weights={},
                layer_class=torch.broadcast_to,
                op_call_args=[params.other_expand_shape],
                op_call_kwargs={},
                functional_op=torch.broadcast_to
            )
            graph.add_node(other_expand_node)
        else:
            other_expand_node = None

        return [input_expand_node, other_expand_node]

    @staticmethod
    def _reshape_input(graph: Graph,
                       matmul_node: FunctionalNode,
                       params: MatMulParams) -> List[FunctionalNode]:
        """
        This method creates the nodes that reshape the input nodes to be 4D before the split.

        Args:
            graph: Graph to apply the substitution on.
            matmul_node: MatMul node.
            params: MatMul shape params.

        Returns:
            Input & Other reshape nodes.
        """
        input_reshape_node = FunctionalNode(
            name=f'{matmul_node.name}_input_reshape',
            framework_attr={},
            input_shape=params.input_expand_shape,
            output_shape=params.input_reshape_shape,
            weights={},
            layer_class=torch.reshape,
            op_call_args=[params.input_reshape_shape],
            op_call_kwargs={},
            functional_op=torch.reshape
        )
        other_reshape_node = FunctionalNode(
            name=f'{matmul_node.name}_other_reshape',
            framework_attr={},
            input_shape=params.other_expand_shape,
            output_shape=params.other_reshape_shape,
            weights={},
            layer_class=torch.reshape,
            op_call_args=[params.other_reshape_shape],
            op_call_kwargs={},
            functional_op=torch.reshape
        )
        # Add reshapes to graph
        if params.prev_input_node:
            graph.add_node_with_in_edges(input_reshape_node, [params.prev_input_node])
        else:
            graph.add_node(input_reshape_node)

        if params.prev_other_node:
            graph.add_node_with_in_edges(other_reshape_node, [params.prev_other_node])
        else:
            graph.add_node(other_reshape_node)

        return [input_reshape_node, other_reshape_node]

    @staticmethod
    def _split_inputs(graph: Graph,
                      matmul_node: FunctionalNode,
                      params: MatMulParams) -> List[FunctionalNode]:
        """
        This method creates the nodes that split the parameters from 4D to 3D for single MatMul operations.

        Args:
            graph: Graph to apply the substitution on.
            matmul_node: MatMul node.
            params: MatMul shape params.

        Returns:
            Input & Other unbind nodes - output of each is list of 3D matrices
        """
        input_split_node = FunctionalNode(
            name=f'{matmul_node.name}_input_split',
            framework_attr={},
            input_shape=params.input_reshape_shape,
            output_shape=params.input_split_shape,
            weights={},
            layer_class=torch.unbind,
            op_call_args=[1],
            op_call_kwargs={},
            functional_op=torch.unbind
        )

        other_split_node = FunctionalNode(
            name=f'{matmul_node.name}_other_split',
            framework_attr={},
            input_shape=params.other_reshape_shape,
            output_shape=params.other_split_shape,
            weights={},
            layer_class=torch.unbind,
            op_call_args=[1],
            op_call_kwargs={},
            functional_op=torch.unbind
        )

        if params.prev_input_node:
            graph.add_node_with_in_edges(input_split_node, [params.prev_input_node])
        else:
            graph.add_node(input_split_node)
        if params.prev_other_node:
            graph.add_node_with_in_edges(other_split_node, [params.prev_other_node])
        else:
            graph.add_node(other_split_node)

        return [input_split_node, other_split_node]

    @staticmethod
    def _calc_single_matmul(graph: Graph,
                            matmul_node: FunctionalNode,
                            input_split_node: FunctionalNode,
                            other_split_node: FunctionalNode,
                            dim_index: int,
                            params: MatMulParams) -> FunctionalNode:
        """
        This method creates the per channel (index) matmul.
        Retrieves the matrices from index dim_index and multiplies them.

        Args:
            graph: Graph to apply the substitution on.
            matmul_node: Original Matmul node
            input_split_node: input after reshape and split.
            other_split_node: other after reshape and split.
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
            input_shape=params.input_split_shape,
            output_shape=params.input_matmul_shape,
            weights={},
            layer_class=operator.getitem,
            op_call_args=[dim_index],
            op_call_kwargs={},
            functional_op=operator.getitem
        )
        graph.add_node_with_in_edges(get_input_node, [input_split_node], [dim_index])
        # Get the other in index dim_index
        get_other_node = FunctionalNode(
            name=f'{matmul_node.name}_other_{dim_index}',
            framework_attr={},
            input_shape=params.other_split_shape,
            output_shape=params.other_matmul_shape,
            weights={},
            layer_class=operator.getitem,
            op_call_args=[dim_index],
            op_call_kwargs={},
            functional_op=operator.getitem
        )
        graph.add_node_with_in_edges(get_other_node, [other_split_node], [dim_index])

        matmul_node = FunctionalNode(name=f'{matmul_node.name}_matmul_{dim_index}',
                                     framework_attr={},
                                     input_shape=(params.input_matmul_shape, params.other_matmul_shape),
                                     output_shape=[params.single_matmul_shape],
                                     weights={},
                                     layer_class=torch.matmul,
                                     op_call_args=[],
                                     op_call_kwargs={},
                                     functional_op=torch.matmul)
        graph.add_node_with_in_edges(matmul_node, [get_input_node, get_other_node])

        return matmul_node

    @staticmethod
    def _stack_matmul_outputs(graph: Graph,
                              matmul_node: FunctionalNode,
                              split_matmul_nodes: List[FunctionalNode],
                              params: MatMulParams) -> FunctionalNode:
        """
        This method creates the node that concats all single matmuls together and then reshapes to the original output
        shape.

        Args:
            graph: Graph to apply the substitution on.
            matmul_node: Original Matmul node
            split_matmul_nodes: list of all single matmul nodes.
            params: MatMul Params

        Returns:
            Node after reshape - final output
        """
        # [(B, m, n)] * (D_1*...*D_N) --> (B, (D_1*...*D_N), m, n)
        cat_node = FunctionalNode(
            name=f'{matmul_node.name}_stack',
            framework_attr={DIM: 1},
            input_shape=[params.single_matmul_shape] * params.matmul_stack_shape[1],
            output_shape=params.matmul_stack_shape,
            weights={},
            layer_class=torch.stack,
            op_call_args=[],
            op_call_kwargs={DIM: 1},
            functional_op=torch.stack,
            inputs_as_list=True
        )
        graph.add_node_with_in_edges(cat_node, split_matmul_nodes)

        if params.matmul_stack_shape != params.output_shape:
            # (B, (D_1 * ... * D_N), m, n) --> (B, D_1, ..., D_N, m, n)
            matmul_reshape_node = FunctionalNode(
                name=f'{matmul_node.name}_reshape',
                framework_attr={},
                input_shape=params.matmul_stack_shape,
                output_shape=params.output_shape,
                weights={},
                layer_class=torch.reshape,
                op_call_args=[params.output_shape],
                op_call_kwargs={},
                functional_op=torch.reshape
            )
            graph.add_node_with_in_edges(matmul_reshape_node, [cat_node])

        return matmul_reshape_node if params.matmul_stack_shape != params.output_shape else cat_node

    @staticmethod
    def _connect_to_graph(
            graph: Graph,
            matmul_node: FunctionalNode,
            head_input_node: FunctionalNode,
            head_other_node: FunctionalNode,
            output_node: FunctionalNode):
        """
        Connect the subgraph to the input graph.
        Args:
            graph: input graph
            matmul_node: MatMul node to substitute inputs and outputs with
            head_input_node: 1st input to MatMul Node
            head_other_node: 2nd input to MatMul Node
            output_node: output node of decomposed MatMul.
        """
        input_in_edge, other_in_edge = graph.in_edges(matmul_node)
        if graph.get_edge_data(*input_in_edge, 0).get('sink_index') == 0:
            graph.add_edge(input_in_edge[0], head_input_node, **graph.get_edge_data(*input_in_edge, 0))
            graph.add_edge(other_in_edge[0], head_other_node, **graph.get_edge_data(*other_in_edge, 0))
        else:
            graph.add_edge(input_in_edge[0], head_other_node, **graph.get_edge_data(*input_in_edge, 0))
            graph.add_edge(other_in_edge[0], head_input_node, **graph.get_edge_data(*other_in_edge, 0))
        graph.remove_edge(input_in_edge[0], matmul_node)
        graph.remove_edge(other_in_edge[0], matmul_node)
        graph.reconnect_out_edges(current_node=matmul_node, new_node=output_node)
