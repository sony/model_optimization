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
from torch import reshape
import torch

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.pytorch.constants import BATCH_DIM_VALUE


class RemoveIdentity(common.BaseSubstitution):
    """
    Remove `torch.nn.Identity` layers from the graph.
    """

    def __init__(self):
        nodes = NodeOperationMatcher(torch.nn.Identity)
        super().__init__(matcher_instance=nodes)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        The method to perform the substitution of the `torch.nn.Identity` node by
        reconnecting its input directly to its output, effectively removing the node
        from the graph.

        Args:
            graph: The current graph of operations where the node resides.
            node: The specific `BaseNode` that is matched to be an Identity operation.

        Returns:
            Graph: The updated graph after removing the identity node.
        """

        # Retrieve the predecessor nodes of the identity node.
        prev_identity_nodes = graph.get_prev_nodes(node)
        # Ensure there is exactly one predecessor; otherwise, do nothing.
        if len(prev_identity_nodes) != 1:
            return graph

        # Reconnect the output edges of the identity node to its predecessor,
        # effectively bypassing the identity node.
        graph.reconnect_out_edges(current_node=node, new_node=prev_identity_nodes[0])
        # Remove the edge from the predecessor to the identity node.
        graph.remove_edge(prev_identity_nodes[0], node)
        # Remove the identity node from the graph.
        graph.remove_node(node_to_remove=node)

        return graph

