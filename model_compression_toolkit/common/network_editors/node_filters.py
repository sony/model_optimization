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

from typing import Any
from model_compression_toolkit.common.matchers.node_matcher import BaseNodeMatcher

class NodeTypeFilter(BaseNodeMatcher):
    """
    Class NodeNameFilter to check if a node is of a specific type.
    """
    def __init__(self, node_type):
        """
        Init a NodeTypeFilter object.

        Args:
            node_type: Node type to check.
        """
        self.node_type = node_type

    def apply(self, input_object: Any) -> bool:
        """
        Check if input_object is of the type that NodeTypeFilter contains.

        Args:
            input_object: Node object to check for its type.

        Returns:
            True if the node if of the type that was passed during the initialization of NodeTypeFilter.
        """
        if input_object.layer_class == self.node_type:
            return True


class NodeNameFilter(BaseNodeMatcher):
    """
    Class NodeNameFilter to check if a node's name has a specific value.
    """
    def __init__(self, node_name):
        """
        Init a NodeNameFilter object.

        Args:
            node_name: Node name to check.
        """
        self.node_name = node_name

    def apply(self, input_object: Any) -> bool:
        """
        Check if input_object's node name is the name that NodeNameFilter contains.

        Args:
            input_object: Node object to check for its name.

        Returns:
            True if the node's name is tha same as the name that was passed during the initialization of NodeNameFilter.
        """
        if input_object.name == self.node_name:
            return True


class NodeNameScopeFilter(BaseNodeMatcher):
    """
    Class NodeNameFilter to check if a string is in a node's name.
    """
    def __init__(self, node_name_scope):
        """
        Init a NodeNameScopeFilter object.

        Args:
            node_name_scope: String to check if exists in node name.
        """
        self.node_name_scope = node_name_scope

    def apply(self, input_object: Any) -> bool:
        """
        Check if input_object's node name contains the string NodeNameScopeFilter contains.

        Args:
            input_object: Node object to check its name contains the string.

        Returns:
            True if the node's name contains the string that was passed during the initialization of NodeNameScopeFilter.
        """
        if self.node_name_scope in input_object.name:
            return True
