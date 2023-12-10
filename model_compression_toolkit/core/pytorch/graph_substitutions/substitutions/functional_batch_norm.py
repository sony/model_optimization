# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.pytorch.constants import *
from model_compression_toolkit.logger import Logger


class FunctionalBatchNorm(common.BaseSubstitution):
    """
    Replace functional batch_norm with BatchNorm2d.
    """

    def __init__(self):
        """
        Matches: functional batch_norm
        """
        bn_node = NodeOperationMatcher(F.batch_norm)
        super().__init__(matcher_instance=bn_node)

    def get_attributes_from_inputs(self, graph: Graph, node: BaseNode) -> dict:
        input_nodes = graph.get_prev_nodes(node, sink_index_sorted=True)

        if len(input_nodes) == 5:
            return {
                MOVING_MEAN: list(input_nodes[1].weights.values())[0],
                MOVING_VARIANCE: list(input_nodes[2].weights.values())[0],
                GAMMA: list(input_nodes[3].weights.values())[0],
                BETA: list(input_nodes[4].weights.values())[0]
            }
        else:
            Logger.warning(f'functional batch_norm is only folded in the 5 inputs case (input, mean, var, gamma, beta),'
                           f'got {len(input_nodes)}')
            return {}

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        Substitute functional.batch_norm and its inputs with BatchNorm2d.
        Args:
            graph: Graph we apply the substitution on.
            node: node that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """
        # if the input is not a 4D tensor, we can't substitute it with BatchNorm2d
        if len(node.input_shape[0]) != 4:
            return graph
        out_channels = node.output_shape[0][1]

        bn_node_weights = self.get_attributes_from_inputs(graph, node)
        if not bn_node_weights:
            return graph
        new_batchnorm2d = BaseNode(name=node.name + '_into_BatchNorm2d',
                                   framework_attr={NUM_FEATURES: out_channels,
                                                   EPSILON: EPSILON_VAL,
                                                   MOMENTUM: MOMENTUM_VAL},
                                   input_shape=node.output_shape,
                                   output_shape=node.output_shape,
                                   weights=bn_node_weights,
                                   layer_class=nn.BatchNorm2d)

        num_nodes_before_substitution = len(graph.nodes)
        num_edges_before_substitution = len(graph.edges)

        batch_norm_consts = graph.get_prev_nodes(node)[1:]
        for const in batch_norm_consts:
            graph.remove_edge(const, node)
            graph.remove_node(const)

        graph.replace_node(node, new_batchnorm2d)

        assert num_nodes_before_substitution - len(graph.nodes) == len(batch_norm_consts)
        assert num_edges_before_substitution - len(graph.edges) == len(batch_norm_consts)

        return graph
