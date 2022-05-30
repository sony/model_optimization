# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

from abc import ABC
from typing import List, Any

from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.matchers import node_matcher, base_graph_filter, edge_matcher
from model_compression_toolkit.core.common.matchers.walk_matcher import WalkMatcherList


class GraphSearches(base_graph_filter.BaseGraphFilter, ABC):
    """
    Apply searches on graphs.
    The graph needs to have 'nodes' and 'edges' attributes, and a 'get_next_nodes' method.
    """

    def _node_filter(self, node_matcher: node_matcher.BaseNodeMatcher) -> list:
        """
        Iterate over nodes and returns the nodes in the graph that matches the matcher object.

        Args:
            node_matcher: Matcher object to apply on nodes in the graph.

        Returns:
            List of nodes that match the node_matcher.
        """

        return [n for n in self.nodes if node_matcher.apply(n)]

    def _edge_filter(self, edge_matcher: edge_matcher.BaseEdgeMatcher) -> list:
        """
        Iterate over edges and returns the edges in the graph that matches
        the edge_matcher object.

        Args:
            edge_matcher: Matcher object to apply on edge.

        Returns:
            List of edges that match.
        """

        edge_list = []
        for e in self.edges:
            if edge_matcher.apply(e) and len(self.edges(e[0])):
                edge_list.append(e)

        return edge_list

    def _walk_filter(self, walk_matcher: WalkMatcherList) -> List[BaseNode]:
        """
        Search for a list of nodes which match the list in walk_matcher.
        If one the nodes in the list (that was found in the graph) has more than one output,
        this list is not returned.

        Args:
            walk_matcher: WalkMatcherList with a list of nodes to match.

        Returns:
            A list of nodes which match the list in walk_matcher.
        """

        def walk_match(node: BaseNode,
                       node_list: List[BaseNode],
                       index: int,
                       node_matcher_list: list) -> Any:
            """
            Iterate the nodes in the graph starting from 'node', and search for its next
            nodes that matches the node next to 'node' in the node_matcher_list.
            If such a node is found, keep searching from that node, by calling walk_match recursively.

            Args:
                node: Node to check if it and its next nodes matches the list in node_matcher_list.
                node_list: A list of nodes that were found so far.
                index: Index of 'node' in the list on nodes it searches.
                node_matcher_list: List of nodes to search for.

            Returns:
                The list on nodes, if found. Otherwise, None.
            """
            if node_matcher_list[index].apply(node):
                node_list.append(node)
                if (index + 1) == len(node_matcher_list):
                    return [node_list]
                result_list = [
                    walk_match(nn, node_list.copy(), index + 1, node_matcher_list) for
                    nn in self.get_next_nodes(node) if
                    # Exclude patterns with an intermediate node with multiple outputs. If it's the last
                    # node in the matcher list, it is a valid pattern and should be checked.
                    len(self.get_next_nodes(nn)) == 1 or (index + 2) == len(node_matcher_list)]
                result_filter = [r for r_list in result_list if r_list is not None for r in r_list if
                                 r is not None and len(r) == len(node_matcher_list)]
                if len(result_filter) == 1:
                    return result_filter
                elif len(result_filter) == 0:
                    return None
                else:  # not supported yet
                    return result_filter
            else:
                return None

        matcher_list = walk_matcher.matcher_list if isinstance(walk_matcher, WalkMatcherList) else [
            walk_matcher]
        result = []

        # Walk the entire graph, node by node
        result_match_list = [walk_match(n, [], 0, matcher_list) for n in self.nodes if len(self.get_next_nodes(n)) == 1]
        # Flatten lists
        result.extend([r for r_list in result_match_list if r_list is not None for r in r_list])
        return result
