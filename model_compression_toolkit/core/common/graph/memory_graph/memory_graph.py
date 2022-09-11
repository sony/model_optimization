from typing import List, Tuple, Any
import numpy as np
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX, EDGE_SOURCE_INDEX
from model_compression_toolkit.core.common.graph.memory_graph.bipartite_graph import DirectedBipartiteGraph


class ActivationMemoryTensor:

    def __init__(self, shape: Tuple[Any], node_name: str, node_output_index: int):

        # TODO: verify that batch size is actually being passed here
        # remove batch size (first element) from output shape
        self.shape = [s[1:] for s in shape]
        self.total_size = self._get_tensor_total_size()

        self.node_name = node_name
        self.node_output_index = node_output_index

    def _get_tensor_total_size(self):
        assert all([x is not None for x in self.shape])
        return np.prod(self.shape)


class MemoryGraph(DirectedBipartiteGraph):

    def __init__(self, model_graph: Graph):

        # TODO: here we need to include all graph nodes, even nodes that don't have activation to quantize,
        #  since their output might be some quantized node input.
        #  The actual memory of a CUT need to be calculated later based on the actual
        #  nodes that have activation to quantize.

        nodes = list(model_graph.nodes)
        memory_tensors = []
        node_to_tensor = []
        tensor_to_node = []

        for n in nodes:
            n_outputs = [n.output_shape] if isinstance(n.output_shape, tuple) else n.output_shape
            out_edges = model_graph.out_edges(n, sort_by_attr=EDGE_SOURCE_INDEX)

            for i, ot in enumerate(n_outputs):
                memory_tensor = ActivationMemoryTensor(ot, n.name, i)
                memory_tensors.append(memory_tensor)
                # Add memory tensor as current node's output
                node_to_tensor.append((n, memory_tensor))

                ot_edges = [oe for oe in out_edges if oe.source_index == i]
                for oe in ot_edges:
                    # Add current memory tensor as input to current node's successors
                    tensor_to_node.append((memory_tensor, oe.sink_node))

        #
        # node_to_memory_tensor = {n: ActivationMemoryTensor(n.output_shape, n.name) for n in model_graph.nodes}
        #
        # nodes = list(node_to_memory_tensor.keys())
        # memory_tensors = list(node_to_memory_tensor.values())
        #
        # node_name_to_incoming_nodes = {n.name: [src_node for src_node in model_graph.incoming_edges(n)] for n in nodes}
        # memory_tensor_to_node = {t: n for t in memory_tensors for n in nodes if
        #                          n in node_name_to_incoming_nodes[t.node_name]}
        #
        # super().__init__(name=model_graph.name + "_memory_graph",
        #                  a_nodes=nodes,
        #                  b_nodes=memory_tensors,
        #                  edges_ab=[(n, t) for n, t in node_to_memory_tensor.items()],
        #                  edges_ba=[(t, n) for t, n in memory_tensor_to_node.items()])

        super().__init__(name=model_graph.name + "_memory_graph",
                         a_nodes=nodes,
                         b_nodes=memory_tensors,
                         edges_ab=[nt for nt in node_to_tensor],
                         edges_ba=[tn for tn in tensor_to_node])

        # memory_lbound_single_op is a lower bound for any schedule of the graph.
        # the bound is defined as the maximum of memory requirements out of all operations in the graph
        # (for a single operation the memory requirement is the sum of the memory size of the children and parents)
        nodes_total_memory = [n.get_total_input_params() + n.get_total_output_params() for n in nodes]  # input + output size of each node
        self.memory_lbound_single_op = max(nodes_total_memory)

        self.sources = [n for n in self.a_nodes if len(list(self.predecessors(n))) == 0]  # TODO: this supposed to be the original graph's inputs, maybe add assertion for sanity
        self.sinks = [n for n in self.a_nodes if len(list(self.successors(n))) == 0]  # TODO: this supposed to be the original graph's outputs, maybe add assertion for sanity

    # TODO: add heuristics if necessary
