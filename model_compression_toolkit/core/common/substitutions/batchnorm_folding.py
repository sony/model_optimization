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
import numpy as np
from typing import Tuple, Callable

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import EdgeMatcher, NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode


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
            edge_nodes: Tuple of two nodes (linear op and batchnorm node).

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


class BatchNormalizationForwardFolding(common.BaseSubstitution):
    """
    Fold BatchNormalization or DW-Convolution with kernel 1x1 into subsequent convolution layers with 1x1 kernels.
    """

    def __init__(self,
                 bn_node: NodeOperationMatcher,
                 conv_node: NodeOperationMatcher,
                 update_weights_for_bn_forward_folding_fn: Callable,
                 get_kernel_hw_fn: Callable,
                 is_group_conv_fn: Callable,
                 get_foldable_node_type_and_validity_fn: Callable,
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
        Matches: Batch Normalization (bn node) followed by a Conv node (source node).

        Args:
            bn_node: Node matcher for batch normalization nodes.
            conv_node: Node matcher for convolution type nodes.
            update_weights_for_bn_forward_folding_fn: Function for updating the convolution kernel & bias
                                                      with the batch normalization weights
            get_kernel_hw_fn: Function for getting the kernel height & width shape
            is_group_conv_fn: Function for checking if the linear layer is a group-convolution
            get_foldable_node_type_and_validity_fn: Function for checking whether the node to forward fold is a valid node and it's type
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
        super().__init__(matcher_instance=EdgeMatcher(bn_node, conv_node))
        self.update_weights_for_bn_forward_folding_fn = update_weights_for_bn_forward_folding_fn
        self.get_kernel_hw_fn = get_kernel_hw_fn
        self.is_group_conv_fn = is_group_conv_fn
        self.get_foldable_node_type_and_validity_fn = get_foldable_node_type_and_validity_fn
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
        Fold BatchNormalization into subsequent Convolution layers with 1x1 kernels.

        Args:
            graph: Graph we apply the substitution on.
            edge_nodes: Tuple of two nodes (batchnorm node and linear op).

        Returns:
            Graph after applying the substitution.
        """

        num_nodes_before_substition = len(graph.nodes)
        num_edges_before_substition = len(graph.edges)

        bn_node, conv_node, _ = edge_nodes

        # If the linear operator is part of a reused group (it is the "base" node, or a reused node),
        # we should skip the substitution.
        if conv_node.reuse or conv_node.reuse_group is not None or bn_node.reuse or bn_node.reuse_group is not None:
            return graph

        if len(graph.get_next_nodes(bn_node)) > 1 or len(graph.get_prev_nodes(conv_node)) > 1:
            return graph
        if self.is_group_conv_fn(conv_node):
            return graph
        kernel = conv_node.get_weights_by_keys(self.kernel_str)
        bias = conv_node.get_weights_by_keys(self.bias_str)
        if not np.all(np.array(self.get_kernel_hw_fn(kernel)) == 1):
            return graph
        is_bn, is_dw_valid = self.get_foldable_node_type_and_validity_fn(bn_node)
        if is_bn:
            gamma = bn_node.get_weights_by_keys(self.gamma_str)
            beta = bn_node.get_weights_by_keys(self.beta_str)
            moving_mean = bn_node.get_weights_by_keys(self.moving_mean_str)
            moving_variance = bn_node.get_weights_by_keys(self.moving__variance_str)
            eps = bn_node.framework_attr[self.epsilon_str]
        elif is_dw_valid:
            gamma = bn_node.get_weights_by_keys(self.kernel_str).flatten()
            beta = bn_node.get_weights_by_keys(self.bias_str)
            moving_mean = 0.0
            moving_variance = 1.0
            eps = 0.0
        else:
            return graph

        if gamma is None:
            gamma = 1.0
        if beta is None:
            beta = 0.0
        if bias is None:
            bias = 0.0

        # W * (gamma * (x-mean)/sqrt(var+eps) + bata) + bias ==>  (W * gamma / sqrt()) * X + (bias + W*(beta - gamma*mean/sqrt()))
        weights_scale = gamma / np.sqrt(moving_variance + eps)
        kernel, bias, kernel_name = self.update_weights_for_bn_forward_folding_fn(conv_node, kernel, bias,
                                                                                  weights_scale,
                                                                                  beta - moving_mean * weights_scale)

        framework_attr = copy.copy(conv_node.framework_attr)
        framework_attr[self.use_bias] = True
        if self.layer_name_str is not None:
            framework_attr[self.layer_name_str] = 'bn_' + conv_node.name

        weights_dict = {kernel_name: kernel,
                        self.bias_str: bias}

        conv_bn = copy.deepcopy(conv_node)
        conv_bn_name = 'bn_' + conv_node.name
        conv_bn.name = conv_bn_name
        conv_bn.framework_attr = framework_attr
        conv_bn.weights = weights_dict

        graph.add_node(conv_bn)
        graph.reconnect_out_edges(current_node=conv_node, new_node=conv_bn)
        graph.reconnect_in_edges(current_node=bn_node, new_node=conv_bn)

        graph.replace_output_node(current_node=conv_node, new_node=conv_bn)

        graph.remove_edge(bn_node, conv_node)
        graph.remove_node(bn_node)
        graph.remove_node(conv_node)

        assert num_nodes_before_substition - len(graph.nodes) == 1
        assert num_edges_before_substition - len(graph.edges) == 1
        return graph
