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

from typing import List

import torch

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.constants import THRESHOLD


class ConcatThresholdUpdate(common.BaseSubstitution):
    """
    Find concat layers and match their prior layers thresholds unless prior layer outputs to multiple layers.
    """


    def __init__(self):
        """
        Initialize a threshold_updater object.
        """
        concatination_node = NodeOperationMatcher(torch.cat) | \
            NodeOperationMatcher(torch.concat) 
        super().__init__(matcher_instance=concatination_node)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        Update previous layers thresholds to match concatinations quantization thresholds. No change if
        previous layer outputs to multiple layers. No change in case of uniform quantization. 
        No change in case of multiple quantization candidates (mixed precision).


        Args:
            graph: Graph we apply the substitution on.
            node: Node refference to edit previous nodes thresholds.

        Returns:
            Graph after applying the substitution.
        """
    
        if len(node.candidates_quantization_cfg) == 1 and THRESHOLD in node.candidates_quantization_cfg[0].activation_quantization_cfg.activation_quantization_params:
            concat_threshold = node.candidates_quantization_cfg[0].activation_quantization_cfg.activation_quantization_params[THRESHOLD]
            prev_nodes = graph.get_prev_nodes(node)
            for prev_node in prev_nodes:
                if len(graph.get_next_nodes(prev_node))==1 and prev_node.type != torch.cat and prev_node.type != torch.concat:
                    prev_node.candidates_quantization_cfg[0].activation_quantization_cfg.activation_quantization_params[THRESHOLD] = concat_threshold

        return graph




