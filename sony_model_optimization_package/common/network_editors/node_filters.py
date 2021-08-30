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

from typing import Any
from sony_model_optimization_package.common.matchers.node_matcher import BaseNodeMatcher


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
