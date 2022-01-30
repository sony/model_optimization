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
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d
from typing import Tuple

from model_compression_toolkit import common
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.graph_matchers import EdgeMatcher, NodeOperationMatcher
from model_compression_toolkit.common import BaseNode
from model_compression_toolkit.pytorch.constants import KERNEL, BIAS, USE_BIAS, GAMMA, BETA, EPSILON, MOVING_MEAN, \
    MOVING_VARIANCE


class BatchNormalizationFolding(common.BaseSubstitution):
    """
    Fold BatchNormalization into preceding linear layers.
    """

    def __init__(self):
        """
        Matches: (Conv2D, Conv2DTranspose) -> BatchNormalization.
        """
        bn_node = NodeOperationMatcher(BatchNorm2d)
        source_node = NodeOperationMatcher(Conv2d) | \
                      NodeOperationMatcher(ConvTranspose2d)

        super().__init__(matcher_instance=EdgeMatcher(source_node, bn_node))

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

        kernel = conv_node.get_weights_by_keys(KERNEL)
        bias = conv_node.get_weights_by_keys(BIAS)
        gamma = bn_node.get_weights_by_keys(GAMMA)
        beta = bn_node.get_weights_by_keys(BETA)
        moving_mean = bn_node.get_weights_by_keys(MOVING_MEAN)
        moving_variance = bn_node.get_weights_by_keys(MOVING_VARIANCE)
        eps = bn_node.framework_attr[EPSILON]

        if gamma is None:
            gamma = 1.0
        if beta is None:
            beta = 0.0
        if bias is None:
            bias = 0.0

        weights_scale = gamma / np.sqrt(moving_variance + eps)
        bias = beta + (bias - moving_mean) * weights_scale

        # Update Kernel
        kernel = kernel * weights_scale[:, None, None, None]

        framework_attr = copy.copy(conv_node.framework_attr)
        framework_attr[USE_BIAS] = True

        weights_dict = {KERNEL: kernel,
                        BIAS: bias}

        conv_bn = copy.deepcopy(conv_node)
        conv_bn_name = conv_node.name + '_bn'
        conv_bn.name = conv_bn_name
        conv_bn.framework_attr = framework_attr
        conv_bn.weights = weights_dict

        graph.add_node(conv_bn)
        graph.reconnect_out_edges(current_node=bn_node, new_node=conv_bn)
        graph.reconnect_in_edges(current_node=conv_node, new_node=conv_bn)

        graph.replace_output_node(current_node=bn_node, new_node=conv_bn)

        graph.remove_edge(conv_node, bn_node)
        graph.remove_node(bn_node)
        graph.remove_node(conv_node)

        assert num_nodes_before_substition - len(graph.nodes) == 1
        assert num_edges_before_substition - len(graph.edges) == 1
        return graph
