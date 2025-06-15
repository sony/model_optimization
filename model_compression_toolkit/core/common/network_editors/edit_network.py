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
from typing import List

from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.network_editors import EditRule


def edit_network_graph(graph: Graph,
                       network_editor: List[EditRule]):
    """
    Apply a list of edit rules on a graph.

    Args:
        graph: The graph to edit.
        groups of layers by how they should be quantized, etc.)
        network_editor: List of edit rules to apply to the graph.

    Returns:
        The graph after it has been applied the edit rules from the network editor list.

    """
    for edit_rule in network_editor:
        filtered_nodes = graph.filter(edit_rule.filter)
        for node in filtered_nodes:
            edit_rule.action.apply(node, graph)
    # return graph
