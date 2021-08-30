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


from typing import Any, List

from tensorflow.keras.layers import Layer

from sony_model_optimization_package.common.graph.node import Node
from sony_model_optimization_package.common.matchers import edge_matcher
from sony_model_optimization_package.common.matchers import node_matcher
from sony_model_optimization_package.common.matchers import walk_matcher


class NodeOperationMatcher(node_matcher.BaseNodeMatcher):
    """
    Class NodeOperationMatcher to check if the layer class of a node matches a specific layer.
    """

    def __init__(self, operation: Layer):
        """
        Init for class NodeOperationMathcer.

        Args:
            operation: Which layer to check if matches.
        """

        self.operation = operation

    def apply(self, input_node_object: Any) -> bool:
        """
        Check if input_node_object matches the matcher condition.

        Args:
            input_node_object: Node object to check the matcher on.

        Returns:
            True if input_node_object is the layer the NodeOperationMatcher holds. Otherwise,
            return nothing.
        """

        if input_node_object.layer_class == self.operation:
            return True


class NodeFrameworkAttrMatcher(node_matcher.BaseNodeMatcher):
    """
    Class NodeFrameworkAttrMatcher to check if a node's attribute has a specific value.
    """

    def __init__(self, attr_name: str, attr_value: Any):
        """
        Init a NodeFrameworkAttrMatcher object.

        Args:
            attr_name: Name of node's attribute to check.
            attr_value: Value to check if the attribute is equal to.
        """
        self.attr_name = attr_name
        self.attr_value = attr_value

    def apply(self, input_node_object: Any) -> bool:
        """
        Check if input_node_object has an attribute with the value the NodeFrameworkAttrMatcher
        contains.

        Args:
            input_node_object: Node object to check for its attribute and value.

        Returns:
            True if the node has an attribute with the attribute name and the value that
            were passed during the initialization of NodeFrameworkAttrMatcher.
        """

        if self.attr_name in input_node_object.framework_attr:
            if input_node_object.framework_attr[self.attr_name] == self.attr_value:
                return True


class EdgeMatcher(edge_matcher.BaseEdgeMatcher):
    """
    class EdgeMatcher to check if an edge matches an edge that EdgeMatcher contains.
    """

    def __init__(self, source_matcher: Node, target_matcher: Node):
        """
        Init an EdgeMatcher object.

        Args:
            source_matcher: Source node to match.
            target_matcher: Destination node to match.
        """

        super().__init__(source_matcher, target_matcher)

    def apply(self, input_object: Any) -> bool:
        """
        Check if input_object is a tuple of two nodes and the same nodes that were
        passed during the EdgeMatcher initialization.
        Args:
            input_object: Object to check if equals to the edge EdgeMatcher holds.

        Returns:
            Whether input_object is equal to the edge EdgeMatcher holds or not.
        """

        if isinstance(input_object, tuple) and len(input_object) >= 2:
            return self.source_matcher.apply(input_object[0]) and self.target_matcher.apply(input_object[1])
        else:
            return False


class WalkMatcher(walk_matcher.WalkMatcherList):
    """
    Class WalkMatcher to check if a list of nodes matches another list of nodes.
    """

    def __init__(self, matcher_list: List[Node]):
        """
        Init a WalkMatcher object.

        Args:
            matcher_list: List of nodes to holds for checking.
        """

        super().__init__(matcher_list)

    def apply(self, input_object: Any) -> bool:  # not in use
        """
        Check if a list of nodes matches the list of nodes the WalkMatcher holds.

        Args:
            input_object: Object to check.

        Returns:
            True if input_object matches the list of nodes the WalkMatcher holds.
        """

        pass  # pragma: no cover
