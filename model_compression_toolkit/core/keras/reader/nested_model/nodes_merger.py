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


import copy
from typing import List

from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode


def merge_models_nodes(inner_model_node: BaseNode,
                       outer_graph: Graph,
                       inner_graph: Graph) -> List[BaseNode]:
    """
    Given two MultiDiGraphs (one of an outer model and the second of the inner model), merge their nodes into
    a single nodes list representing the nodes that should be in a single MultiDiGraph after unrolling the inner graph.
    This is done by removing the inner model input nodes and the model node in the outer graph.
    Args:
        outer_graph: MultiDiGraph of the outer model.
        inner_graph: MultiDiGraph of the inner model.
        inner_model_node: Node of the inner model in the graph of the outer model.

    Returns:
        List of nodes that should be in a single graph after unrolling the inner graph.
    """
    # merge internal nodes
    res_nodes = copy.copy(list(outer_graph.nodes))
    res_nodes.extend(inner_graph.nodes)
    for input_node in inner_graph.get_inputs():  # inputs of inner model will no longer be needed.
        res_nodes.remove(input_node)
    # Remove the inner model node since we unroll it, and it's not going to be a node in the final graph.
    res_nodes.remove(inner_model_node)
    return res_nodes
