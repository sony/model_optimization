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
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.graph.edge import EDGE_SOURCE_INDEX
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

        super().__init__(name=model_graph.name + "_memory_graph",
                         a_nodes=nodes,
                         b_nodes=memory_tensors,
                         edges_ab=[nt for nt in node_to_tensor],
                         edges_ba=[tn for tn in tensor_to_node])

        # memory_lbound_single_op is a lower bound for any schedule of the graph.
        # the bound is defined as the maximum of memory requirements out of all operations in the graph
        # (for a single operation the memory requirement is the sum of the memory size of the children and parents)
        inputs_tensors_memory = [sum([t.total_size for t in self.operation_node_children(n)])
                                 for n in nodes if n in model_graph.get_inputs()]

        nodes_total_memory = [sum([t.total_size for t in self.operation_node_children(n)] +
                                  [t.total_size for t in self.operation_node_parents(n)])
                              for n in nodes if n not in model_graph.get_inputs()]

        self.memory_lbound_single_op = max(nodes_total_memory + inputs_tensors_memory)

        self.sources_a = [n for n in self.a_nodes if len(list(self.predecessors(n))) == 0]

        # Note that unlike the original scheduler,
        # we don't need sinks_a since we assume all layers have activation tensor.
        # In oppose to them we do need sinks_b to allow creating single target for astar in case of multiple outputs.
        self.sinks_b = [n for n in self.b_nodes if len(list(self.successors(n))) == 0]
        assert len([n for n in self.a_nodes if len(list(self.successors(n))) == 0]) == 0, \
            "All operations should have an activation tensor, so there are no supposed to be sink nodes in side A," \
            "of the bipartite memory graph."

    def update_sources_a(self):
        self.sources_a = [n for n in self.a_nodes if len(list(self.predecessors(n))) == 0]

    def update_sinks_b(self):
        self.sinks_b = [n for n in self.b_nodes if len(list(self.successors(n))) == 0]

    def activation_tensor_children(self, activation_tensor):
        return [oe[1] for oe in self.out_edges(activation_tensor)]

    def activation_tensor_parents(self, activation_tensor):
        return [ie[0] for ie in self.in_edges(activation_tensor)]

    def operation_node_children(self, op_node):
        return [oe[1] for oe in self.out_edges(op_node)]

    def operation_node_parents(self, op_node):
        return [ie[0] for ie in self.in_edges(op_node)]

