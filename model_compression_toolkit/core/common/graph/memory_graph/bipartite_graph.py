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

from typing import Any, List, Tuple
import networkx as nx


class DirectedBipartiteGraph(nx.DiGraph):

    def __init__(self,
                 name: str,
                 a_nodes: List[Any],
                 b_nodes: List[Any],
                 edges_ab: List[Tuple[Any, Any]],
                 edges_ba: List[Tuple[Any, Any]],
                 **attr):
        super().__init__(**attr)

        self.name = name

        self.a_nodes = set()
        self.b_nodes = set()
        self._update_nodes_sets()

        self.add_nodes_from(a_nodes, bipartite=0)
        self.add_nodes_from(b_nodes, bipartite=1)

        # TODO: do we need to check edges validity or is it enforced by adding the 0-1 flag to the nodes?
        self.add_edges_from(edges_ab)
        self.add_edges_from(edges_ba)

    def _update_nodes_sets(self):
        self.a_nodes = {n for n, d in self.nodes(data=True) if d["bipartite"] == 0}
        self.b_nodes = set(self) - self.a_nodes

    def add_nodes_to_a(self, new_nodes):
        self.add_nodes_from(new_nodes, bipartite=0)
        self._update_nodes_sets()

    def add_nodes_to_b(self, new_nodes):
        self.add_nodes_from(new_nodes, bipartite=1)
        self._update_nodes_sets()

    def add_edges(self, new_edges):
        self.add_edges_from(new_edges)
