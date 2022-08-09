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

from typing import List

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode


class SoftmaxShift(common.BaseSubstitution):
    """
    Shift layer by const before Softmax
    """

    def __init__(self,
                 nodes: List[BaseNode],
                 bias_str: str):
        """
        Matches: (Softmax)

        Args:
            nodes: Nodes matcher for Linear/Dense - softmax type nodes.
            bias_str: The framework specific attribute name of the convolution layer's bias.
        """
        super().__init__(matcher_instance=nodes)
        self.bias_str = bias_str

    def substitute(self,
                   graph: Graph,
                   nodes: List[BaseNode]) -> Graph:
        """
        Shift the layer before Softmax activation.

        Args:
            graph: Graph we apply the substitution on.
            nodes: nodes that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """

        first_node = nodes[0]

        if first_node.is_activation_quantization_enabled():
            tensor_stat = graph.get_out_stats_collector(first_node)
            if isinstance(tensor_stat, common.StatsCollector):
                max_value = tensor_stat.mpcc.max
                min_value = tensor_stat.mpcc.min
                shift_value = -1 * (max_value + min_value) / 2

                if first_node.get_weights_by_keys(self.bias_str) is not None:
                    b1 = first_node.get_weights_by_keys(self.bias_str)
                else:
                    b1 = 0.0
                b1_fixed = b1 + shift_value
                first_node.set_weights_by_keys(self.bias_str, b1_fixed)
                graph.shift_stats_collector(first_node, shift_value)

        return graph
