# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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


from typing import Any, List

from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.matchers import node_matcher, walk_matcher, edge_matcher


class NodeOperationMatcher(node_matcher.BaseNodeMatcher):
    """
    Class NodeOperationMatcher to check if the layer class of a node matches a specific layer.
    """

    def __init__(self, operation: Any):
        """
        Init for class NodeOperationMathcer.

        Args:
            operation: Which layer to check if matches.
        """

        self.operation = operation

    def apply(self, input_node_object: BaseNode) -> bool:
        """
        Check if input_node_object matches the matcher condition.

        Args:
            input_node_object: Node object to check the matcher on.

        Returns:
            True if input_node_object is the layer the NodeOperationMatcher holds. Otherwise,
            return nothing.
        """

        if input_node_object.is_match_type(self.operation):
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

    def __init__(self, source_matcher: BaseNode, target_matcher: BaseNode):
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

    def __init__(self, matcher_list: List[BaseNode]):
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
