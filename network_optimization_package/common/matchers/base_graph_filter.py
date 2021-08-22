# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================

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
