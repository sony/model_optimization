from typing import List, Tuple, Any
import numpy as np
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.graph.memory_graph.bipartite_graph import DirectedBipartiteGraph


class ActivationMemoryTensor:

    def __init__(self, shape: Tuple[Any], node_name: str, id: int):

        # TODO: verify that batch size is actually being passed here
        # remove batch size (first element) from output shape
        self.shape = [s[1:] for s in shape]
        self.total_size = self._get_tensor_total_size()

        self.node_name = node_name
        self.id = id


    def _get_tensor_total_size(self):
        assert all([x is not None for x in self.shape])
        return np.prod(self.shape)


class MemoryGraph(DirectedBipartiteGraph):

    def __init__(self, model_graph: Graph):

        # TODO: here we need to include all graph nodes, even nodes that don't have activation to quantize,
        #  since their output might be some quantized node input.
        #  The actual memory of a CUT need to be calculated later based on the actual
        #  nodes that have activation to quantize.

        # TODO: take care of multiple outputs - how do we know which output tensor goes for which node?
        #  also need to keep as list of tuples not dict (since multiple key can occur).
        #  Consider starting with list of tuples of size 3 (node, tensor, node) and split it later to the required lists

        # TODO: add 'id' to tensoors creation
        node_to_memory_tensor = {n: ActivationMemoryTensor(n.output_shape, n.name) for n in model_graph.nodes}

        nodes = list(node_to_memory_tensor.keys())
        memory_tensors = list(node_to_memory_tensor.values())

        node_name_to_incoming_nodes = {n.name: [src_node for src_node in model_graph.incoming_edges(n)] for n in nodes}
        memory_tensor_to_node = {t: n for t in memory_tensors for n in nodes if
                                 n in node_name_to_incoming_nodes[t.node_name]}

        super().__init__(name=model_graph.name + "_memory_graph",
                         a_nodes=nodes,
                         b_nodes=memory_tensors,
                         edges_ab=[(n, t) for n, t in node_to_memory_tensor.items()],
                         edges_ba=[(t, n) for t, n in memory_tensor_to_node.items()])

        # memory_lbound_single_op is a lower bound for any schedule of the graph
        # the bound is defined as the maximum of memory requirements out of all operations in the graph
        # (for a single operation the memory requirement is the sum of the memory size of the children and parents)
        # TODO: understand and implement calculation
        self.memory_lbound_single_op
