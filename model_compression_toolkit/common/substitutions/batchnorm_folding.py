# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
import numpy as np
from typing import Tuple, Callable

from model_compression_toolkit import common
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.graph_matchers import EdgeMatcher, NodeOperationMatcher
from model_compression_toolkit.common.graph.base_node import BaseNode


class BatchNormalizationFolding(common.BaseSubstitution):
    """
    Fold BatchNormalization into preceding linear layers.
    """

    def __init__(self,
                 source_node: NodeOperationMatcher,
                 bn_node: NodeOperationMatcher,
                 update_kernel_for_bn_folding_fn: Callable,
                 kernel_str: str,
                 bias_str: str,
                 gamma_str: str,
                 beta_str: str,
                 moving_mean_str: str,
                 moving__variance_str: str,
                 epsilon_str: str,
                 use_bias: str,
                 layer_name_str: str):
        """
        Matches: Conv node (source node) to Batch Normalization (bn node).

        Args:
            source_node: Node matcher for convolution type nodes.
            bn_node: Node matcher for batch normalization nodes.
            update_kernel_for_bn_folding_fn: Function for updating the convolution kernel
            with the batch normalization weights
            kernel_str: The framework specific attribute name of the convolution layer's weight/kernel.
            bias_str: The framework specific attribute name of the convolution layer's bias.
            gamma_str: The framework specific attribute name of the batch norm layer's gamma parameter.
            beta_str: The framework specific attribute name of the batch norm layer's beta parameter.
            moving_mean_str: The framework specific attribute name of the batch norm layer's moving mean parameter.
            moving__variance_str: The framework specific attribute name of the batch norm layer's moving variance parameter.
            epsilon_str: The framework specific attribute name of the batch norm layer's epsilon parameter.
            use_bias: The framework specific attribute name of the convolution layer's bias flag.
            layer_name_str: The framework specific attribute name of layer's name.
        """
        super().__init__(matcher_instance=EdgeMatcher(source_node, bn_node))
        self.update_kernel_for_bn_folding_fn = update_kernel_for_bn_folding_fn
        self.kernel_str = kernel_str
        self.bias_str = bias_str
        self.gamma_str = gamma_str
        self.beta_str = beta_str
        self.moving_mean_str = moving_mean_str
        self.moving__variance_str = moving__variance_str
        self.epsilon_str = epsilon_str
        self.use_bias = use_bias
        self.layer_name_str = layer_name_str

    def substitute(self,
                   graph: Graph,
                   edge_nodes: Tuple[BaseNode, BaseNode]) -> Graph:
        """
        Fold BatchNormalization into preceding linear layers.

        Args:
            graph: Graph we apply the substitution on.
            edge_nodes: Tuple of tow nodes (linear op and batchnorm node).

        Returns:
            Graph after applying the substitution.
        """

        num_nodes_before_substition = len(graph.nodes)
        num_edges_before_substition = len(graph.edges)

        conv_node = edge_nodes[0]

        # If the linear operator is part of a reused group (it is the "base" node, or a reused node),
        # we should skip the substitution.
        if conv_node.reuse or conv_node.reuse_group is not None:
            return graph

        bn_node = edge_nodes[1]

        if len(graph.get_next_nodes(conv_node)) > 1 or len(graph.get_prev_nodes(bn_node)) > 1:
            return graph

        kernel = conv_node.get_weights_by_keys(self.kernel_str)
        bias = conv_node.get_weights_by_keys(self.bias_str)
        gamma = bn_node.get_weights_by_keys(self.gamma_str)
        beta = bn_node.get_weights_by_keys(self.beta_str)
        moving_mean = bn_node.get_weights_by_keys(self.moving_mean_str)
        moving_variance = bn_node.get_weights_by_keys(self.moving__variance_str)
        eps = bn_node.framework_attr[self.epsilon_str]

        if gamma is None:
            gamma = 1.0
        if beta is None:
            beta = 0.0
        if bias is None:
            bias = 0.0

        weights_scale = gamma / np.sqrt(moving_variance + eps)
        bias = beta + (bias - moving_mean) * weights_scale

        kernel, kernel_name = self.update_kernel_for_bn_folding_fn(conv_node, kernel, weights_scale)

        framework_attr = copy.copy(conv_node.framework_attr)
        framework_attr[self.use_bias] = True
        if self.layer_name_str is not None:
            framework_attr[self.layer_name_str] = conv_node.name + '_bn'

        weights_dict = {kernel_name: kernel,
                        self.bias_str: bias}

        conv_bn = copy.deepcopy(conv_node)
        conv_bn_name = conv_node.name + '_bn'
        conv_bn.name = conv_bn_name
        conv_bn.framework_attr = framework_attr
        conv_bn.weights = weights_dict

        graph.add_node(conv_bn)
        graph.reconnect_out_edges(current_node=bn_node, new_node=conv_bn)
        graph.reconnect_in_edges(current_node=conv_node, new_node=conv_bn)

        graph.replace_output_node(current_node=bn_node, new_node=conv_bn)

        conv_bn.prior_info = bn_node.prior_info

        graph.remove_edge(conv_node, bn_node)
        graph.remove_node(bn_node)
        graph.remove_node(conv_node)

        assert num_nodes_before_substition - len(graph.nodes) == 1
        assert num_edges_before_substition - len(graph.edges) == 1
        return graph
