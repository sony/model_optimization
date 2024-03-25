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

import itertools
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common import BaseNode, Graph, BaseSubstitution
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualSplitWeightsNode, \
    VirtualSplitActivationNode
from model_compression_toolkit.core.common.matchers.base_matcher import BaseMatcher


class BaseWeightsActivationSplit(BaseSubstitution):
    def __init__(self,
                 activation_layer_type: type,
                 fw_attr: dict,
                 matcher_instance: BaseMatcher):

        self.activation_layer_type = activation_layer_type
        self.fw_attr = fw_attr
        super().__init__(matcher_instance=matcher_instance)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        Decompose a linear node into two nodes - one with the linear operations (a weights node) and one with
        the activation operation (with an identity function that just passes the node's output, but allows to
        quantize it according to the node's activation quantization configuration candidates).
        The two new virtual nodes are connected with an edge [weights node --> activation node].
        Note that the node is split only if its candidates list is composite, that is, it contains all the combinations
        of activation and weights bit-width that exists in any of its candidates.

        Args:
            graph: Graph we apply the substitution on.
            node: Node to split.

        Returns:
            Graph after applying the substitution.
        """
        # The decomposition works on linear nodes, that is, nodes with kernel ops
        kernel_attr = graph.fw_info.get_kernel_op_attributes(node.type)[0]
        if kernel_attr is None:
            Logger.error(f"Trying to split node weights and activation, but node "
                         f"{node.name} doesn't have a kernel attribute.")
        if not node.is_all_weights_candidates_equal(kernel_attr) and not node.is_all_activation_candidates_equal():
            # Node has both different weights and different activation configuration candidates
            weights_bits = [c.weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits
                            for c in node.get_unique_weights_candidates(kernel_attr)]
            activation_bits = [c.activation_quantization_cfg.activation_n_bits for c in node.get_unique_activation_candidates()]
            expected_candidates = list(itertools.product(weights_bits, activation_bits))
            all_candidates_bits = [(c.weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits,
                                    c.activation_quantization_cfg.activation_n_bits)
                                   for c in node.candidates_quantization_cfg]
            if not set(expected_candidates).issubset(all_candidates_bits):
                # Node is not composite, therefore, can't be split
                Logger.critical(f"The node {node.name} cannot be split as it has non-composite candidates. "
                                f"For mixed-precision search with BOPS target resource utilization, "
                                f"all model layers must be composite.")  # pragma: no cover

        weights_node = VirtualSplitWeightsNode(node, kernel_attr)
        activation_node = VirtualSplitActivationNode(node, self.activation_layer_type, self.fw_attr)

        # Update graph
        graph.add_node(weights_node)
        graph.add_node(activation_node)
        graph.reconnect_in_edges(current_node=node, new_node=weights_node)
        graph.reconnect_out_edges(current_node=node, new_node=activation_node)
        graph.replace_output_node(current_node=node, new_node=activation_node)
        graph.add_edge(weights_node,
                       activation_node,
                       source_index=0,
                       sink_index=0)
        graph.remove_node(node)

        return graph

