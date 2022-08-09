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

import numpy as np
import tensorflow as tf
if tf.__version__ < "2.6":
    from tensorflow.keras.layers import LayerNormalization, BatchNormalization
else:
    from keras.layers import LayerNormalization, BatchNormalization

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.constants import REUSE, REUSE_GROUP
from model_compression_toolkit.core.keras.constants import MOVING_VARIANCE, MOVING_MEAN, BETA, \
    GAMMA, EPSILON, AXIS, CENTER, SCALE


class LayerNormDecomposition(common.BaseSubstitution):
    """
    Adds a BatchNorm node after a LayerNormalization node in the graph that applies the LayerNorm's
    gamma & beta weights in the BatchNorm
    """

    def __init__(self):
        """
        Matches LayerNorm node.
        """
        super().__init__(matcher_instance=NodeOperationMatcher(LayerNormalization))

    @staticmethod
    def _get_weight_by_name(_node, w_str):
        """
        get weight by part of weight name
        :param _node: node to search weights in
        :param w_str: part of name to search
        :return: matching weight
        """
        return [k for k in _node.weights.keys() if w_str in k][0]

    def substitute(self,
                   graph: Graph,
                   ln_node: BaseNode) -> Graph:
        """
        Adds a BatchNorm node after a LayerNormalization node in the graph that applies the LayerNorm's
        gamma & beta weights in the BatchNorm

        Args:
            graph: Graph we apply the substitution on.
            ln_node: LayerNorm node to append with BatchNorm

        Returns:
            Graph after applying the substitution.
        """

        # if both scale and center are False (no gamma & beta) the substitution isn't required
        if ln_node.framework_attr[SCALE] is False and ln_node.framework_attr[CENTER] is False:
            return graph

        bn_node_name = f'{ln_node.name}_bn'
        bn_weights = {}

        # get gamma from LayerNorm, if exists
        if ln_node.framework_attr[SCALE]:
            gamma_name = self._get_weight_by_name(ln_node, GAMMA)
            gamma = ln_node.weights[gamma_name].copy()
        else:
            gamma = np.ones((ln_node.input_shape[-1],))
        bn_weights.update({f'{bn_node_name}/{GAMMA}:': gamma})

        # get beta from LayerNorm, if exists
        if ln_node.framework_attr[CENTER]:
            beta_name = self._get_weight_by_name(ln_node, BETA)
            beta = ln_node.weights[beta_name].copy()
        else:
            beta = np.zeros((ln_node.input_shape[-1],))
        bn_weights.update({f'{bn_node_name}/{BETA}:': beta})

        # Set BatchNorm mean & variance to 0 & 1 respectively (normalization done in LayerNorm)
        bn_weights.update({f'{bn_node_name}/{MOVING_MEAN}:': np.zeros((ln_node.input_shape[-1],))})
        bn_weights.update({f'{bn_node_name}/{MOVING_VARIANCE}:': np.ones((ln_node.input_shape[-1],))})

        epsilon = ln_node.framework_attr[EPSILON]
        axis = ln_node.framework_attr[AXIS]

        _reuse_params = {REUSE: ln_node.reuse, REUSE_GROUP: ln_node.reuse_group}
        new_ln_node = BaseNode(ln_node.name, {AXIS: axis, EPSILON: epsilon, CENTER: False, SCALE: False},
                               ln_node.input_shape, ln_node.output_shape, {}, LayerNormalization,
                               **_reuse_params)
        graph.add_node(new_ln_node)
        bn_node = BaseNode(bn_node_name, {AXIS: axis, EPSILON: 0},
                           ln_node.input_shape, ln_node.output_shape, bn_weights, BatchNormalization,
                           **_reuse_params)
        graph.add_node_with_in_edges(bn_node, [new_ln_node])

        # Connect new nodes
        _in_edge = list(graph.in_edges(ln_node))[0]
        graph.add_edge(_in_edge[0], new_ln_node, **graph.get_edge_data(*_in_edge, 0))
        graph.remove_edge(_in_edge[0], ln_node)
        graph.reconnect_out_edges(current_node=ln_node, new_node=bn_node)

        # Finally, remove the LayerNorm node
        graph.remove_node(ln_node, new_graph_outputs=[OutTensor(bn_node, 0)])

        return graph
