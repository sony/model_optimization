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
from typing import Callable

import numpy as np

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.target_platform import QuantizationMethod


class BatchNormalizationReconstruction(common.BaseSubstitution):
    """
    Reconstruct BatchNormalization after linear layers.
    """
    def __init__(self,
                 source_node: NodeOperationMatcher,
                 create_bn_node: Callable,
                 gamma_str: str,
                 beta_str: str,
                 moving_mean_str: str,
                 moving_variance_str: str,
                 epsilon_val: float):
        """
        Matches: Source (linear) nodes.

        Args:
            source_node: Node matcher for linear type nodes.
            create_bn_node: Function for creating BN nodes.
            gamma_str: The framework specific attribute name of the batch norm layer's gamma parameter.
            beta_str: The framework specific attribute name of the batch norm layer's beta parameter.
            moving_mean_str: The framework specific attribute name of the batch norm layer's moving mean parameter.
            moving_variance_str: The framework specific attribute name of the batch norm layer's moving
            variance parameter.
            epsilon_val: The framework specific attribute value of the batch norm layer's epsilon parameter.
        """
        super().__init__(matcher_instance=source_node)
        self.create_bn_node = create_bn_node
        self.gamma_str = gamma_str
        self.beta_str = beta_str
        self.moving_mean_str = moving_mean_str
        self.moving_variance_str = moving_variance_str
        self.epsilon_val = epsilon_val

    def substitute(self,
                   graph: Graph,
                   source_node: BaseNode) -> Graph:
        """
        Reconstruct BatchNormalization after linear layers.

        Args:
            graph: Graph we apply the substitution on.
            source_node: linear type node.

        Returns:
            Graph after applying the substitution.
        """

        num_nodes_before_substitution = len(graph.nodes)
        num_edges_before_substitution = len(graph.edges)

        # We apply only on nodes with folded BatchNormalization.
        if source_node.prior_info.std_output is None or source_node.prior_info.mean_output is None:
            for qc in source_node.candidates_quantization_cfg:
                qc.weights_quantization_cfg.weights_second_moment_correction = False
            return graph

        # This feature disabled for models with weights quantization method of Power of 2
        for qc in source_node.candidates_quantization_cfg:
            if qc.weights_quantization_cfg.weights_quantization_method == QuantizationMethod.POWER_OF_TWO:
                Logger.warning("Second moment statistics correction feature disabled for models with weights "
                               "quantization method of Power of 2")
                for qc_inner in source_node.candidates_quantization_cfg:
                    qc_inner.weights_quantization_cfg.weights_second_moment_correction = False
                return graph

        eps = self.epsilon_val

        original_gamma = source_node.prior_info.std_output
        beta = source_node.prior_info.mean_output
        moving_mean = beta
        moving_var = np.power(original_gamma, 2)
        gamma = np.sqrt(moving_var + eps)

        bn_node_weights = {self.gamma_str: gamma,
                           self.beta_str: beta,
                           self.moving_mean_str: moving_mean,
                           self.moving_variance_str: moving_var}

        bn_node = self.create_bn_node(source_node, bn_node_weights)

        bn_node.prior_info = copy.deepcopy(source_node.prior_info)

        bn_node.candidates_quantization_cfg = copy.deepcopy(source_node.candidates_quantization_cfg)

        for qc in bn_node.candidates_quantization_cfg:
            qc.weights_quantization_cfg.enable_weights_quantization = False
            qc.activation_quantization_cfg.enable_activation_quantization = False

        graph.reconnect_out_edges(current_node=source_node, new_node=bn_node)
        graph.replace_output_node(current_node=source_node, new_node=bn_node)
        graph.add_node_with_in_edges(bn_node, [source_node])

        assert len(graph.nodes) - num_nodes_before_substitution == 1
        assert len(graph.edges) - num_edges_before_substitution == 1

        return graph
