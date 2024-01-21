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
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.pytorch.constants import *
from model_compression_toolkit.logger import Logger


class FunctionalLayerNorm(common.BaseSubstitution):
    """
    Replace functional layer_norm with LayerNorm.
    """

    def __init__(self):
        """
        Matches: functional layer_norm
        """
        ln_node = NodeOperationMatcher(F.layer_norm)
        super().__init__(matcher_instance=ln_node)

    def get_attributes_from_inputs(self, graph: Graph, node: BaseNode, normalized_shape: [Tuple, List, int]) -> dict:
        """
        Parse layer_norm(input, normalized_shape, weight=None, bias=None)
        Args:
            graph: Graph we apply the substitution on.
            node: Node that match the pattern in the substitution init.
            normalized_shape: nn.LayerNorm "normalized_shape" argument

        Returns:
            Graph after applying the substitution.
        """

        # Get input nodes (sorted)
        input_nodes = graph.get_prev_nodes(node, sink_index_sorted=True)

        # Define default weight and bias
        w0 = np.ones(normalized_shape) # Default value in case weight is not given
        b0 = np.zeros(normalized_shape) # Default value in case bias is not given

        # Check if weight and/or bias were not given.
        has_weight = WEIGHT not in node.framework_attr
        has_bias = BIAS not in node.framework_attr

        weight_input_ind = 1 if has_weight else 0
        bias_input_ind = weight_input_ind + 1

        return {
            GAMMA: list(input_nodes[weight_input_ind].weights.values())[0] if has_weight else w0,
            BETA: list(input_nodes[bias_input_ind].weights.values())[0] if has_bias else b0
        }

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        Substitute functional.layer_norm and its inputs with LayerNorm.
        Args:
            graph: Graph we apply the substitution on.
            node: node that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """
        normalized_shape = node.input_shape[0][-1]

        ln_node_weights = self.get_attributes_from_inputs(graph, node, normalized_shape)

        new_layernorm = BaseNode(name=node.name + '_into_LayerNorm',
                                 framework_attr={NORMALIZED_SHAPE: normalized_shape,
                                                 EPSILON: node.framework_attr.get('eps'),
                                                 },
                                 input_shape=node.output_shape,
                                 output_shape=node.output_shape,
                                 weights=ln_node_weights,
                                 layer_class=nn.LayerNorm)

        num_nodes_before_substitution = len(graph.nodes)
        num_edges_before_substitution = len(graph.edges)

        layer_norm_consts = graph.get_prev_nodes(node)[1:]
        for const in layer_norm_consts:
            graph.remove_edge(const, node)
            graph.remove_node(const)

        graph.replace_node(node, new_layernorm)

        assert num_nodes_before_substitution - len(graph.nodes) == len(layer_norm_consts)
        assert num_edges_before_substitution - len(graph.edges) == len(layer_norm_consts)

        return graph
