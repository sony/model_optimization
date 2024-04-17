from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode


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
        return graph

    # Reconnect the output edges of the identity node to its predecessor,
    # effectively bypassing the identity node.
    graph.reconnect_out_edges(current_node=node, new_node=prev_identity_nodes[0])
    # Remove the edge from the predecessor to the identity node.
    graph.remove_edge(prev_identity_nodes[0], node)
    # Remove the identity node from the graph.
    graph.remove_node(node_to_remove=node)

    return graph
