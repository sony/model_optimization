# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.layer_filter_params import LayerFilterParams


def filter_fusing_patterns(fusing_patterns: List[List[Any]], node: BaseNode, idx: int = 0) -> List[List[Any]]:
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
            if (type(fusing_pattern[idx]) == LayerFilterParams and node.is_match_filter_params(fusing_pattern[idx])) or fusing_pattern[idx] == node.type:
                valid_fusing_patterns.append(fusing_pattern)

    # Return only valid patterns for this node
    return valid_fusing_patterns


def is_valid_fusion(fusing_patterns: List[List[Any]], nodes: List[BaseNode]) -> bool:
    """
    Check if the fusion is valid: exist in fusing_patterns
    Args:
        fusing_patterns: supported fusings
        nodes: nodes which are participating in fusion
    Returns:
        whether the fusion in valid
    """
    fusion_depth = len(nodes)
    if fusion_depth <= 1:
        return False
    for fusing_pattern in fusing_patterns:
        if fusion_depth != len(fusing_pattern):
            continue
        counter = 0
        for i,layer in enumerate(fusing_pattern):
            if (type(layer) == LayerFilterParams and nodes[i].is_match_filter_params(layer)) or layer == nodes[i].type:
                counter += 1
        if counter == fusion_depth:
            return True
    return False


def disable_nodes_activation_quantization(nodes: List[BaseNode]):
    """
    Disable activation for non-quantization needed due to fusion
    Args:
        nodes: nodes to update their activation quantization
    """
    for node in nodes:
        for qc in node.candidates_quantization_cfg:
            qc.activation_quantization_cfg.enable_activation_quantization = False


def fusion(graph: Graph, tpc: TargetPlatformCapabilities) -> Graph:
    """
    Fusing defines a list of operators that should be combined and treated as a single operator,
    hence no quantization is applied between them when they appear in the graph.
    This function search and disable quantization for such patterns.
    Args:
        graph: Graph we apply the fusion on.
        tpc: TargetPlatformCapabilities object that describes the desired inference target platform (includes fusing patterns MCT should handle).
    Returns:
        Graph after applying fusion activation marking.
    """
    fusing_patterns = tpc.get_fusing_patterns()
    if len(fusing_patterns) == 0:
        return graph

    # Find max fusion
    max_layers_fusing = 0 if len(fusing_patterns) == 0 else max([len(fusing_pattern) for fusing_pattern in fusing_patterns])


    # -------------------------------- #
    # Fusion algorithm
    # -------------------------------- #
    fused_graph = copy.deepcopy(graph)

    # Travel along the graph to find layers for fusing
    nodes = fused_graph.get_topo_sorted_nodes()
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
            next_nodes = fused_graph.get_next_nodes(fusing_nodes[-1])
            if len(next_nodes) != 1:  # Give up if node has more than one connection (not supported for fusion)
                break

        # New fusion: mark all nodes in the fusion except last one
        if is_valid_fusion(fusing_patterns, fusing_nodes):
            fused_nodes.extend(fusing_nodes)
            disable_nodes_activation_quantization(fusing_nodes[:-1])
            fused_graph.update_fused_nodes(fusing_nodes)

    return fused_graph
