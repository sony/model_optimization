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
import copy
from typing import Any, List
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.common.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.common.target_platform.targetplatform2framework.layer_filter_params import LayerFilterParams

def filter_fusing_patterns(fusing_patterns: List[List[Any]], node: BaseNode, idx: int = 0):
    """
    Update relevant fusing patterns object if layer number 'idx' inside the fusion matches the node
    Args:
        fusing_patterns: supported fusings
        node: node to decide if it can be a part of fusion
        idx: index of layer in the fusion
    Returns:
        fusing_patterns after filtering non-relevant fusions
    """
    valid_fusing_patterns = []
    for i,fusing_pattern in enumerate(fusing_patterns):
        if idx < len(fusing_pattern):
            if (type(fusing_pattern[idx]) == LayerFilterParams and fusing_pattern[idx].match(node)) or fusing_pattern[idx] == node.type:
                valid_fusing_patterns.append(fusing_pattern)

    # Return only valid patterns for this node
    return valid_fusing_patterns


def mark_fusing_activation(nodes: List[BaseNode]):
    """
    Mark activation for non-quantization needed due to fusion
    Args:
        nodes: nodes to update their activation quantization
    """
    for node in nodes:
        for qc in node.candidates_quantization_cfg:
            qc.activation_quantization_cfg.enable_activation_quantization = False


def fusion(graph: Graph, tpc: TargetPlatformCapabilities) -> Graph:
    """
    There are cases (layers fusion) we do not need to quantize a layer's output since there is an
    activation layer that follows it. Thus, in these cases we set the node's attribute
    that indicates whether the output should be quantized to False.
    tpc (TargetPlatformCapabilities) holds the fusing information.
    This function searches and marks activation for no quantization if fusion is found.
    Args:
        graph: Graph we apply the fusion on.
        tpc: TargetPlatformCapabilities object which holds fusion configuration
    Returns:
        Graph after applying fusion activation marking.
    """
    fusing_patterns = tpc.get_fusing_patterns()
    if len(fusing_patterns) == 0:
        return graph

    max_layers_fusing = 0
    for fusing_pattern in fusing_patterns:
        num_layers_fusing = len(fusing_pattern)
        max_layers_fusing = max(num_layers_fusing, max_layers_fusing)

    # -------------------------------- #
    # Fusion algorithm
    # -------------------------------- #
    # Travel along the graph to find layers for fusing
    nodes = graph.get_topo_sorted_nodes()
    fused_nodes = []  # nodes that are participating in fusing
    for node in nodes:
        # Skip if already in fusing
        if node in fused_nodes:
            continue
        # Start fusing search
        fusing_nodes = [] # nodes that are candidates for participating in fusing
        patterns = copy.deepcopy(fusing_patterns)
        next_nodes = [node]
        for i in range(max_layers_fusing):
            patterns = filter_fusing_patterns(patterns, next_nodes[0], i)
            if len(patterns) == 0: # Give up if no more fusion pattern
                break
            fusing_nodes.append(next_nodes[0])
            next_nodes = graph.get_next_nodes(fusing_nodes[-1])
            if len(next_nodes) != 1:  # Give up if node has more than one connection (not supported for fusion)
                break

        fused_nodes.extend(fusing_nodes)

        # New fusion: mark all nodes in the fusion except last one
        if len(fusing_nodes) > 1:
            mark_fusing_activation(fusing_nodes[:-1])
            graph.user_info.add_fusion(fusing_nodes)

    return graph
