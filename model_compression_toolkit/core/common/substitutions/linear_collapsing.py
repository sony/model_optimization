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
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import EdgeMatcher, NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode


class Conv2DCollapsing(common.BaseSubstitution):
    """
    Collapse Conv2D into preceding Conv2D (Not non-linear activation between them)
    """
    def __init__(self,
                 first_node: NodeOperationMatcher,
                 second_node: NodeOperationMatcher,
                 conv2d_collapsing_fn: Callable,
                 kernel_str: str,
                 kernel_size: str,
                 bias_str: str,
                 use_bias_str: str,
                 strides_str: str,
                 paddings_str: str,
                 dilations_str: str,
                 groups_str: str,
                 filters_str: str,
                 data_format_str: str = None,
                 layer_name_str: str = None):
        """
        Collapsing Conv2D node (first node) to Conv2D node (second node).
        Args:
            first_node: Node matcher for convolution type nodes.
            second_node: Node matcher for convolution type nodes.
            conv2d_collapsing_fn: Function for updating the convolution kernel and bias
            kernel_str: The framework specific attribute name of the convolution layer's weight/kernel.
            kernel_size_str: The framework specific attribute name of the convolution layer's kernel size.
            bias_str: The framework specific attribute name of the convolution layer's bias.
            use_bias_str: The framework specific attribute name of the convolution layer's bias flag.
            strides_str: The framework specific attribute name of the convolution layer's strides.
            paddings_str: The framework specific attribute name of the convolution layer's paddings.
            dilations_str: The framework specific attribute name of the convolution layer's dilations.
            groups_str: The framework specific attribute name of the convolution layer's groups.
            filters_str: The framework specific attribute name of the convolution layer's filters.
            data_format_str: The framework specific attribute name of the convolution layer's data format.
            layer_name_str: The framework specific attribute name of layer's name.
        """
        super().__init__(matcher_instance=EdgeMatcher(first_node, second_node))
        self.conv2d_collapsing_fn = conv2d_collapsing_fn
        self.kernel_str = kernel_str
        self.kernel_size = kernel_size
        self.bias_str = bias_str
        self.use_bias_str = use_bias_str
        self.strides_str = strides_str
        self.paddings_str = paddings_str
        self.dilations_str = dilations_str
        self.groups_str = groups_str
        self.filters_str = filters_str
        self.data_format_str = data_format_str
        self.layer_name_str = layer_name_str

    def substitute(self,
                   graph: Graph,
                   edge_nodes: Tuple[BaseNode, BaseNode]) -> Graph:
        """
        Collapse linear layer into preceding linear layers.
        Convolution condition:
        |----------------------------|      |--------------|
        | Conv2D k1xk2 -> Conv2D 1x1 | ---> | Conv2D k1xk2 |
        |----------------------------|      |--------------|
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

        # If there is an extra connection between these two nodes skip the substitution
        if len(graph.get_next_nodes(first_node)) > 1 or len(graph.get_prev_nodes(second_node)) > 1:
            return graph

        # Skip if convolution's data format is 'channels_first'
        if self.data_format_str is not None:
            data_format1 = first_node.framework_attr[self.data_format_str]
            data_format2 = second_node.framework_attr[self.data_format_str]
            if data_format1 == 'channels_first' or data_format2 == 'channels_first':
                Logger.warning("No supported Conv2D with 'channels_first' in block collapsing!")
                return graph

        # Get kernels and strides
        kernel2_size = second_node.framework_attr[self.kernel_size]
        paddings2 = second_node.framework_attr[self.paddings_str]
        strides1 = first_node.framework_attr[self.strides_str]
        strides2 = second_node.framework_attr[self.strides_str]
        dilations1 = first_node.framework_attr[self.dilations_str]
        dilations2 = second_node.framework_attr[self.dilations_str]
        groups1 = first_node.framework_attr[self.groups_str]
        groups2 = second_node.framework_attr[self.groups_str]

        # Check if convolutions parameters satisfy with the collapsing conditions, if not skip the
        # Collapsing 2 linear convolutions can be done only when:
        # 1. No strides and dilations
        # 2. First kernel is (k1,k2), second kernel is (1,1)
        # 3. No convolution groups
        # 4. No padding between the convolutions (padding2):
        #   a. Keras: 1x1 is always zero padding for padding2='same' or 'valid'
        #   b. Pytorch: padding2 can be 'same' or 'valid', and also can be padding=(p1,p2). for the latter we support only padding=(0,0).
        skip = True
        if kernel2_size == (1,1) and strides1 == strides2 == dilations1 == dilations2 == (1,1) \
                and groups1 == groups2 == 1 and (isinstance(paddings2, str) or paddings2 == (0,0)):
            skip = False
        if skip:
            return graph

        # New collapsed weights
        kernel, bias = self.conv2d_collapsing_fn(first_node, second_node, self.kernel_str, self.bias_str)

        # If the number of parameters is bigger after the collapsing, skip the substitution
        kernel1 = first_node.get_weights_by_keys(self.kernel_str)
        kernel2 = second_node.get_weights_by_keys(self.kernel_str)
        num_parameters = np.prod(kernel1.shape) + np.prod(kernel2.shape)
        new_num_parameters = np.prod(kernel.shape)
        if num_parameters < new_num_parameters:
            return graph

        num_nodes_before_substition = len(graph.nodes)
        num_edges_before_substition = len(graph.edges)

        # New collapsed node
        conv_collapsed = copy.deepcopy(first_node)
        conv_collapsed_name = first_node.name + '_' + second_node.name + '_collapsed'
        conv_collapsed.name = conv_collapsed_name
        weights_dict = {self.kernel_str: kernel}
        conv_collapsed.framework_attr[self.use_bias_str] = False
        if bias is not None:
            weights_dict.update({self.bias_str: bias})
            conv_collapsed.framework_attr[self.use_bias_str] = True
        conv_collapsed.weights = weights_dict
        conv_collapsed.input_shape = first_node.input_shape
        conv_collapsed.framework_attr[self.filters_str] = second_node.framework_attr[self.filters_str]
        if self.layer_name_str is not None:
            conv_collapsed.framework_attr[self.layer_name_str] = conv_collapsed_name

        # Update graph
        graph.add_node(conv_collapsed)
        graph.reconnect_out_edges(current_node=second_node, new_node=conv_collapsed)
        graph.reconnect_in_edges(current_node=first_node, new_node=conv_collapsed)
        graph.replace_output_node(current_node=second_node, new_node=conv_collapsed)

        graph.remove_edge(first_node, second_node)
        graph.remove_node(first_node)
        graph.remove_node(second_node)

        # Sanity check
        assert num_nodes_before_substition - len(graph.nodes) == 1
        assert num_edges_before_substition - len(graph.edges) == 1

        return graph
