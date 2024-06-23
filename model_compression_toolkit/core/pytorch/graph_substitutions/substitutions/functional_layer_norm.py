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
from typing import Dict, Tuple, List

from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
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

    @staticmethod
    def get_attributes_from_weights(node: FunctionalNode, normalized_shape: [Tuple, List, int]) -> Dict:
        """
        Convert functional layer_norm positional weights to LayerNorm weights. Extract indices of gamma
        and beta according to tensor_input_allocs if they were input as kwargs. If they were input as args,
        use their fixed positions.
        Args:
            node: Node that match the pattern in the substitution init.
            normalized_shape: nn.LayerNorm "normalized_shape" argument

        Returns:
            Weights dictionary for LayerNorm.
        """

        # Define default weight and bias
        weights_dict = {GAMMA: np.ones(normalized_shape),  # Default value in case weight is not given
                        BETA: np.zeros(normalized_shape)  # Default value in case bias is not given
                        }

        # Check if weight and/or bias were not given.
        if KERNEL in node.tensor_input_allocs:
            weights_dict[GAMMA] = node.weights[node.tensor_input_allocs.index(KERNEL)]
        elif KERNEL not in node.op_call_kwargs:
            weights_dict[GAMMA] = node.weights[1]

        if BIAS in node.tensor_input_allocs:
            weights_dict[BETA] = node.weights[node.tensor_input_allocs.index(BIAS)]
        elif BIAS not in node.op_call_kwargs:
            weights_dict[BETA] = node.weights[2]

        return weights_dict

    def substitute(self,
                   graph: Graph,
                   node: FunctionalNode) -> Graph:
        """
        Substitute functional.layer_norm and its inputs with LayerNorm.
        Args:
            graph: Graph we apply the substitution on.
            node: node that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """
        normalized_shape = node.input_shape[0][-1]

        ln_node_weights = self.get_attributes_from_weights(node, normalized_shape)

        framework_attr = {NORMALIZED_SHAPE: normalized_shape}
        if EPSILON in node.op_call_kwargs:
            framework_attr.update({EPSILON: node.op_call_kwargs[EPSILON]})
        new_layernorm = BaseNode(name=node.name + '_into_LayerNorm',
                                 framework_attr=framework_attr,
                                 input_shape=node.output_shape,
                                 output_shape=node.output_shape,
                                 weights=ln_node_weights,
                                 layer_class=nn.LayerNorm)

        graph.replace_node(node, new_layernorm)
        return graph
