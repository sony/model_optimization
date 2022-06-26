# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from torch import permute
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode


class PermuteCallMethod(common.BaseSubstitution):
    """
    Find "permute" node to substitute new dimension argument if needed
    """

    def __init__(self):
        """
        Matches: 'permute' node
        """
        nodes = NodeOperationMatcher(permute)
        super().__init__(matcher_instance=nodes)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        Wrap dimension of permute with tuple if it's missing

        Args:
            graph: Graph we apply the substitution on.
            node: node that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """
        # Check op_call_args is not empty and has its argument as a tuple
        if node.op_call_args and not isinstance(node.op_call_args[0], tuple):
            node.op_call_args = [node.op_call_args]
        return graph
