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
from typing import Dict
import numpy as np
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

    @staticmethod
    def get_attributes_from_weights(node: BaseNode) -> Dict:
        """
        convert functional batch_norm positional weights to BatchNorm2d weights
        Args:
            node: functional batch_norm node.

        Returns:
            Weights dictionary for BatchNorm2d.
        """
        if 1 not in node.weights and 2 not in node.weights:
            Logger.critical(f'Missing {MOVING_MEAN} and {MOVING_VARIANCE} in functional batch_norm inputs.')
        weights_dict = {MOVING_MEAN: node.weights[1],
                        MOVING_VARIANCE: node.weights[2],
                        GAMMA: np.ones(node.weights[1].shape),
                        BETA: np.zeros(node.weights[1].shape)}

        has_weight = WEIGHT not in node.framework_attr
        has_bias = BIAS not in node.framework_attr

        if 3 in node.weights:
            if has_weight:
                weights_dict[GAMMA] = node.weights[3]
            else:
                weights_dict[BETA] = node.weights[3]
        if 4 in node.weights:
            assert has_bias
            weights_dict[BETA] = node.weights[4]

        return weights_dict

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

        bn_node_weights = self.get_attributes_from_weights(node)
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

        graph.replace_node(node, new_batchnorm2d)
        return graph
