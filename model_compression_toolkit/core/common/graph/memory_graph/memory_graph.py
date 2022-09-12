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

from typing import List, Tuple, Any
import numpy as np
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX, EDGE_SOURCE_INDEX
from model_compression_toolkit.core.common.graph.memory_graph.bipartite_graph import DirectedBipartiteGraph
from model_compression_toolkit.core.common.graph.memory_graph.memory_element import ActivationMemoryTensor


class MemoryGraph(DirectedBipartiteGraph):

    def __init__(self, model_graph: Graph):

        # TODO: here we need to include all graph nodes, even nodes that don't have activation to quantize,
        #  since their output might be some quantized node input.
        #  The actual memory of a CUT need to be calculated later based on the actual
        #  nodes that have activation to quantize.

        self.model_graph = model_graph

        nodes = list(model_graph.nodes)
        memory_tensors = []
        node_to_tensor = []
        tensor_to_node = []

        output_nodes = [output.node for output in model_graph.get_outputs()]
        for n in nodes:
            if n in output_nodes:
                # We don't add the output tensor of an output node (since it's not needed to be saved in memory).
                # Also, if we do add its tensor than we won't have sink nodes.
                # TODO: verify with Alon that this is the expected behavior.
                continue
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

        self.sources_a = [n for n in self.a_nodes if len(list(self.predecessors(n))) == 0]
        self.sinks_a = [n for n in self.a_nodes if len(list(self.successors(n))) == 0]

    def update_sources_a(self):
        self.sources_a = [n for n in self.a_nodes if len(list(self.predecessors(n))) == 0]

    def update_sinks_a(self):
        self.sinks_a = [n for n in self.a_nodes if len(list(self.successors(n))) == 0]

    def activation_tensor_children(self, activation_tensor):
        return [oe[1] for oe in self.out_edges(activation_tensor)]

    def activation_tensor_parents(self, activation_tensor):
        return [ie[0] for ie in self.in_edges(activation_tensor)]

    def operation_node_children(self, op_node):
        return [oe[1] for oe in self.out_edges(op_node)]

    def operation_node_parents(self, op_node):
        return [ie[0] for ie in self.in_edges(op_node)]

    # def get_op_children_names(self, op_name):
    #     node = self.model_graph.find_node_by_name(op_name)
    #     if len(node) == 0:
    #         return []
    #
    #     assert len(node) == 1, "Node name must be unique"
    #     node = node[0]
    #
    #     return [e.sink_node.name for e in self.model_graph.out_edges(node)]

    # TODO: add heuristics if necessary
