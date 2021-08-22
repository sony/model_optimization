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

from abc import abstractmethod

from . import base_matcher
from . import edge_matcher
from . import function
from . import node_matcher
from . import walk_matcher


class BaseGraphFilter(object):
    """
    Base class to implement graph filtering by nodes, edges and sequences of nodes.
    """

    def filter(self, matcher: base_matcher.BaseMatcher) -> list:
        """
        Receive a matcher and return a list of matches in the graph.

        Args:
            matcher: Object of type BaseMatcher.

        Returns:
            List of matches.
        """

        # Return the nodes that matches the matcher object.
        if function.is_node_matcher(matcher):
            return self._node_filter(matcher)

        # Return the edges that matches the matcher object.
        elif function.is_edge_matcher(matcher):
            return self._edge_filter(matcher)

        # Return a list of nodes that match the matcher.
        elif function.is_walk_matcher(matcher):
            return self._walk_filter(matcher)
        else:
            raise NotImplemented  # pragma: no cover

    @abstractmethod
    def _node_filter(self, node_matcher: node_matcher.BaseNodeMatcher) -> list:
        """
        Returns the nodes in the graph that matches the matcher object.

        Args:
            node_matcher: Matcher object to apply on nodes in the graph.

        Returns:
            List of nodes that match the node_matcher.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _edge_filter(self, edge_matcher: edge_matcher.BaseEdgeMatcher) -> list:
        """
        Returns the edges in the graph that match the matcher object.

        Args:
            edge_matcher: Matcher object to apply on the edges.

        Returns:
            List of edges that match.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _walk_filter(self, walk_matcher: walk_matcher.WalkMatcherList) -> list:
        """
        Search for a list of nodes which match the list in walk_matcher. and return it.
        If one the nodes in the list (that was found in the graph) has more than one output,
        this list is not returned.

        Args:
            walk_matcher: WalkMatcherList with a list of nodes to match.

        Returns:
            A list of nodes which match the list in walk_matcher.
        """
        pass  # pragma: no cover
