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

from model_compression_toolkit.core.common import BaseNode, Graph, BaseSubstitution
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode, \
    VirtualSplitWeightsNode


class BaseVirtualActivationWeightsComposition(BaseSubstitution):
    def __init__(self, matcher_instance):
        super().__init__(matcher_instance=matcher_instance)

    def substitute(self,
                   graph: Graph,
                   weights_node: BaseNode) -> Graph:
        """
        Combines an activation --> weights edge's node into one virtual composed node that contains the activation
        operation and the linear operation (in that order).
        The node's quantization configuration candidates include the cartesian product of both node's candidates.
        Note that if the activation node has multiple outputs (beside its matched weights node) than the substitution
        would not apply.

        Args:
            graph: Graph we apply the substitution on.
            weights_node: A node with linear operation to be combined with its preceding activation.

        Returns:
            Graph after applying the substitution.
        """
        if not isinstance(weights_node, VirtualSplitWeightsNode):
            raise TypeError(f'Matched node {weights_node} was expected to be of type VirtualSplitWeightsNode. '
                            f'This substitution is expected to be called after activation-weights split.')

        predecessors = graph.get_prev_nodes(weights_node)
        assert len(predecessors) == 1, (f'Matched node for {self.__class__.__name__} substitution is expected to have'
                                        f'exactly one input, node {weights_node} has {len(predecessors)}')
        act_node = predecessors[0]
        if len(graph.out_edges(act_node)) > 1:
            Logger.warning(f"Node {act_node.name} has multiple outgoing edges, which is not supported with "
                           f"mixed-precision search under bit-operations constraint. In such case, it might result in "
                           f"incorrect resource utilization computation and suboptimal bits selection.")
            return graph

        # Virtual composed activation-weights node
        # we pass a dummy initialization dict to initialize the super BaseNode class,
        # the actual arguments values are irrelevant because they are being overridden or not used
        v_node = VirtualActivationWeightsNode(act_node,
                                              weights_node,
                                              fw_info=graph.fw_info,
                                              **weights_node.__dict__)

        # Update graph
        graph.add_node(v_node)
        graph.reconnect_in_edges(current_node=act_node, new_node=v_node)
        graph.reconnect_out_edges(current_node=weights_node, new_node=v_node)
        graph.replace_input_node(current_node=act_node, new_node=v_node)
        graph.replace_output_node(current_node=weights_node, new_node=v_node)
        graph.remove_edge(act_node, weights_node)
        graph.remove_node(weights_node)
        graph.remove_node(act_node)

        return graph
