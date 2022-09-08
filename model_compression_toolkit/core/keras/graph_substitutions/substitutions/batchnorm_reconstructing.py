# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D, Conv2DTranspose, Conv2D

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Graph, BaseNode, Logger
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.keras.constants import LINEAR, ACTIVATION, GAMMA, \
    BETA, \
    MOVING_MEAN, MOVING_VARIANCE, EPSILON, MOMENTUM, EPSILON_VAL, MOMENTUM_VAL


class BatchNormalizationReconstructing(common.BaseSubstitution):
    """
    Reconstruct BatchNormalization after linear layers.
    """
    def __init__(self):
        conv_node = NodeOperationMatcher(DepthwiseConv2D) | \
                    NodeOperationMatcher(Conv2D) | \
                    NodeOperationMatcher(Conv2DTranspose)

        activation_linear = NodeFrameworkAttrMatcher(ACTIVATION, LINEAR)
        source_node = conv_node & activation_linear

        super().__init__(matcher_instance=source_node)

    def substitute(self,
                   graph: Graph,
                   source_node: BaseNode) -> Graph:
        """
        Reconstruct BatchNormalization after linear layers.

        Args:
            graph: Graph we apply the substitution on.
            source_node: linear node.

        Returns:
            Graph after applying the substitution.
        """

        num_nodes_before_substitution = len(graph.nodes)
        num_edges_before_substitution = len(graph.edges)

        conv_node = source_node

        # We apply only on nodes with folded BatchNormalization.
        if conv_node.prior_info.std_output is None or conv_node.prior_info.mean_output is None:
            return graph

        # If the linear operator is part of a reused group (it is the "base" node, or a reused node),
        # we should skip the substitution.
        if conv_node.reuse or conv_node.reuse_group is not None:
            return graph

        if len(graph.get_next_nodes(conv_node)) > 1:
            return graph

        # This feature disabled for models with weights quantization method of Power of 2
        for qc in conv_node.candidates_quantization_cfg:
            if qc.weights_quantization_cfg == QuantizationMethod.POWER_OF_TWO:
                Logger.warning("Second moment statistics correction feature disabled for models with weights "
                               "quantization method of Power of 2")
                return graph

        out_node = graph.out_edges(conv_node)[0].sink_node
        source_edge = graph.out_edges(conv_node)[0]

        eps = EPSILON_VAL
        momentum = MOMENTUM_VAL

        original_gamma = conv_node.prior_info.std_output
        beta = conv_node.prior_info.mean_output
        moving_mean = beta
        moving_var = np.power(original_gamma, 2)
        gamma = np.sqrt(moving_var + eps)

        bn_node_weights = {GAMMA: gamma,
                           BETA: beta,
                           MOVING_MEAN: moving_mean,
                           MOVING_VARIANCE: moving_var}

        bn_node = BaseNode(name=conv_node.name + '_reconstructed',
                           framework_attr={EPSILON: eps, MOMENTUM: momentum},
                           input_shape=conv_node.output_shape,
                           output_shape=conv_node.output_shape,
                           weights=bn_node_weights,
                           layer_class=BatchNormalization)
        bn_node.prior_info = copy.deepcopy(conv_node.prior_info)

        bn_node.candidates_quantization_cfg = copy.deepcopy(conv_node.candidates_quantization_cfg)

        for qc in bn_node.candidates_quantization_cfg:
            qc.weights_quantization_cfg.enable_weights_quantization = False
            qc.activation_quantization_cfg.enable_activation_quantization = False

        graph.add_node_with_in_edges(bn_node, [conv_node])
        graph.add_edge(bn_node, out_node, **source_edge.get_attributes())
        graph.remove_edge(conv_node, out_node)

        assert len(graph.nodes) - num_nodes_before_substitution == 1
        assert len(graph.edges) - num_edges_before_substitution == 1
        return graph
