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

from model_compression_toolkit.core.common.graph.base_graph import Graph, OutTensor
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.logger import Logger


def remove_identity_node(graph: Graph,
                         node: BaseNode) -> Graph:
    """
    The method to perform the substitution of the identity node by
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
        # We do not expect to get here.
        Logger.error(f"Identity node {node} have {len(prev_identity_nodes)} inputs, while expected to have one. Skipping remove identity substitution.") # pragma: no cover
        return graph  # pragma: no cover

    graph_outputs = graph.get_outputs()
    for i, g_out in enumerate(graph_outputs):
        if g_out.node == node:
            graph_outputs[i] = OutTensor(node=prev_identity_nodes[0], node_out_index=g_out.node_out_index)

    # Reconnect the output edges of the identity node to its predecessor,
    # effectively bypassing the identity node.
    graph.reconnect_out_edges(current_node=node, new_node=prev_identity_nodes[0])
    # Remove the edge from the predecessor to the identity node.
    graph.remove_edge(prev_identity_nodes[0], node)
    # Remove the identity node from the graph.
    graph.remove_node(node_to_remove=node,
                      new_graph_outputs=graph_outputs
                      )

    return graph
