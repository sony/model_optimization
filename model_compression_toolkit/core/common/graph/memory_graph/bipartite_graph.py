from typing import Any, List, Tuple

import networkx as nx
from networkx.algorithms import bipartite


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

        self.add_nodes_from(a_nodes, bipartite=0)
        self.add_nodes_from(b_nodes, bipartite=1)

        self.a_nodes = {n for n, d in self.nodes(data=True) if d["bipartite"] == 0}
        self.b_nodes = set(self) - self.a_nodes

        # TODO: do we need to check edges validity or is it enforced by adding the 0-1 flag to the nodes?
        self.add_edges_from(edges_ab)
        self.add_edges_from(edges_ba)


# if __name__ == "__main__":
#     g = DirectedBipartiteGraph(
#         name="test",
#         a_nodes=[1, 2, 3, 4],
#         b_nodes=['a', 'b', 'c'],
#         edges_ab=[(1, 'a'), (2, 'b'), (2, 'c')],
#         edges_ba=[('a', 3), ('a', 4), ('c', 4), ('c', 1)]
#     )