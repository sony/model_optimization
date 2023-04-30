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

from model_compression_toolkit.logger import Logger


class DirectedBipartiteGraph(nx.DiGraph):
    """
    Directed Bipartite graph representation.
    """

    def __init__(self,
                 name: str,
                 a_nodes: List[Any],
                 b_nodes: List[Any],
                 edges_ab: List[Tuple[Any, Any]],
                 edges_ba: List[Tuple[Any, Any]],
                 **attr):
        """
        Args:
            name: A name to reference the graph with.
            a_nodes: A set of objects to define as the A-side in the bipartite graph.
            b_nodes: A set of objects to define as the B-side in the bipartite graph.
            edges_ab: Edges from side A to side B.
            edges_ba: Edges from side B to side A.
            **attr: Attributes to add to graph as key=value pairs.
        """

        super().__init__(**attr)

        self.name = name

        self.a_nodes = set()
        self.b_nodes = set()

        self.add_nodes_from(a_nodes, bipartite=0)
        self.add_nodes_from(b_nodes, bipartite=1)
        self._update_nodes_sets()

        # Check edges validity
        self._verify_edges(edges_ab)
        self._verify_edges(edges_ba)

        self.add_edges_from(edges_ab)
        self.add_edges_from(edges_ba)

    def _update_nodes_sets(self):
        """
        Updates the class members of side A and side B nodes.
        """
        self.a_nodes = {n for n, d in self.nodes(data=True) if d["bipartite"] == 0}
        self.b_nodes = set(self) - self.a_nodes

    def _verify_edges(self, edges_list: List[Tuple[Any, Any]]):
        """
        Verifies bipartite correctness of a set of edges to add to the graph.
        If there is an edge in the list that violates the bipartite property - an Exception is raised.

        Args:
            edges_list: A list of edges to verify their correction.
        """
        for n1, n2 in edges_list:
            if n1 in self.a_nodes and n2 in self.a_nodes:
                Logger.critical(f"Can't add an edge {(n1, n2)} between two nodes in size A of a bipartite graph.")
            if n1 in self.b_nodes and n2 in self.b_nodes:
                Logger.critical(f"Can't add an edge {(n1, n2)} between two nodes in size B of a bipartite graph.")

    def add_nodes_to_a(self, new_nodes: List[Any]):
        """
        Add a set of nodes to side A of the bipartite graph.

        Args:
            new_nodes: New nodes to add to side A.
        """
        self.add_nodes_from(new_nodes, bipartite=0)
        self._update_nodes_sets()

    def add_nodes_to_b(self, new_nodes: List[Any]):
        """
        Add a set of nodes to side B of the bipartite graph.

        Args:
            new_nodes: New nodes to add to side B.
        """
        self.add_nodes_from(new_nodes, bipartite=1)
        self._update_nodes_sets()

    def add_edges(self, new_edges: List[Tuple[Any, Any]]):
        """
        Add a set of edges to the bipartite graph.

        Args:
            new_edges: New edges to add to side B.
        """
        self._verify_edges(new_edges)
        self.add_edges_from(new_edges)
