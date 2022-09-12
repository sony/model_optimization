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

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.memory_graph.cut import Cut
from model_compression_toolkit.core.common.graph.memory_graph.memory_element import MemoryElements
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import ActivationMemoryTensor


class DummyBaseNodeGenerator:
    def __init__(self):
        self.counter = 0

    def __iter__(self):
        while True:
            yield BaseNode(name="dummy_node_" + str(self.counter),
                           framework_attr={},
                           input_shape=tuple(),
                           output_shape=tuple(),
                           weights={},
                           layer_class=None,
                           has_activation=False)

            self.counter += 1


class DummyActivationMemoryTensorGenerator:
    def __init__(self):
        self.counter = 0

    def __iter__(self):
        while True:
            yield ActivationMemoryTensor(shape=tuple(),
                                         node_name="dummy_tensor_" + str(self.counter),
                                         node_output_index=0)

            self.counter += 1


# TODO: consider changing the name - it's not a memory graph, it builds a memory graph to rum Astar on its cuts
class CutsMemoryGraph:

    def __init__(self, memory_graph):

        # In order to run Astar search on the graph we need to define a single source and a single target.
        # Dummy nodes are used for this purpose
        # (2 for the src, 4 for the target, memory values are 0 to leave the memory requirement unaffected)
        gen_a = iter(DummyBaseNodeGenerator)
        gen_b = iter(DummyActivationMemoryTensorGenerator)

        # Source Cut
        src_dummy_a = next(gen_a)
        src_dummy_b = next(gen_b)
        edges_src_ab = [(src_dummy_a, src_dummy_b)]
        edges_src_ba = [(src_dummy_b, src_a) for src_a in memory_graph.sources_a]
        src_cut = Cut([src_dummy_a], {src_dummy_a}, MemoryElements(elements={src_dummy_b}, total_size=0))
        # Updated graph sinks - need to connect all sink nodes to a single sink node
        mid_b_nodes = [next(gen_b) for _ in memory_graph.sinks_a]
        sinks_fix_ab_edges = [(original_sink, mid_b) for original_sink, mid_b in zip(memory_graph.sinks_a, mid_b_nodes)]
        sink_fix_node_a = next(gen_a)  # this is the new single sink node
        sinks_fix_ba_edges = [(t, sink_fix_node_a) for s, t in sinks_fix_ab_edges]

        # Target Cut
        target_dummy_a1 = next(gen_a)
        target_dummy_a2 = next(gen_a)
        target_dummy_b1 = next(gen_b)
        target_dummy_b2 = next(gen_b)
        edges_target_ab = [(sink_fix_node_a, target_dummy_b1), (target_dummy_a1, target_dummy_b2)]
        edges_target_ba = [(target_dummy_b1, target_dummy_a1), (target_dummy_b2, target_dummy_a2)]
        target_cut = Cut([], set(), MemoryElements(elements={target_dummy_b1, target_dummy_b2}, total_size=0))

        # Update memory graph
        # New nodes
        self.memory_graph = memory_graph
        self.memory_graph.add_nodes_to_a([src_dummy_a, sink_fix_node_a, target_dummy_a1, target_dummy_a2])
        self.memory_graph.add_nodes_to_b([src_dummy_b, target_dummy_b1, target_dummy_b2] + mid_b_nodes)
        # New Edges
        self.memory_graph.add_edges(edges_src_ab + edges_src_ba + sinks_fix_ba_edges + edges_target_ab + edges_target_ba)
        self.memory_graph.update_sources_a()
        self.memory_graph.update_sinks_a()