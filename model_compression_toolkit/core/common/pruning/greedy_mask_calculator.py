# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

import numpy as np
from typing import List, Dict, Tuple

from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core.common.pruning.mask.per_channel_mask import MaskIndicator
from model_compression_toolkit.core.common.pruning.memory_calculator import MemoryCalculator
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import PruningFrameworkImplementation
from model_compression_toolkit.core.common.pruning.mask.per_simd_group_mask import PerSIMDGroupMask
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities


class GreedyMaskCalculator:
    """
    GreedyMaskCalculator calculates pruning masks for prunable nodes to meet a
    specified target resource utilization. It employs a greedy approach to selectively unprune channel
    groups (SIMD groups) based on their importance scores. Initially, all channels are
    pruned (mask set to zero), and the calculator iteratively adds back the most significant
    channel groups until the memory footprint meets the target resource utilization or all channels are unpruned.
    """
    def __init__(self,
                 prunable_nodes: List[BaseNode],
                 simd_groups_scores: Dict[BaseNode, np.ndarray],
                 target_resource_utilization: ResourceUtilization,
                 graph: Graph,
                 fw_impl: PruningFrameworkImplementation,
                 fqc: FrameworkQuantizationCapabilities,
                 simd_groups_indices: Dict[BaseNode, List[List[int]]]):
        """
        Args:
            prunable_nodes (List[BaseNode]): Nodes that are eligible for pruning.
            simd_groups_scores (Dict[BaseNode, np.ndarray]): Importance scores for each SIMG group in a prunable node.
            target_resource_utilization (ResourceUtilization): The target resource utilization to achieve.
            graph (Graph): The computational graph of the model.
            fw_impl (PruningFrameworkImplementation): Framework-specific implementation details.
            fqc (FrameworkQuantizationCapabilities): Platform-specific constraints and capabilities.
            simd_groups_indices (Dict[BaseNode, List[List[int]]]): Indices of SIMD groups in each node.
        """
        self.prunable_nodes = prunable_nodes
        self.target_resource_utilization = target_resource_utilization
        self.graph = graph
        self.fw_impl = fw_impl
        self.fqc = fqc

        self.simd_groups_indices = simd_groups_indices
        self.simd_groups_scores = simd_groups_scores

        self.oc_pruning_mask = PerSIMDGroupMask(prunable_nodes=prunable_nodes,
                                                simd_groups_indices=simd_groups_indices)

        self.memory_calculator = MemoryCalculator(graph=graph,
                                                  fw_impl=fw_impl)

    def get_mask(self) -> Dict[BaseNode, np.ndarray]:
        """
        Retrieves the current pruning mask for each prunable node.

        Returns:
            Dict[BaseNode, np.ndarray]: The current pruning mask for each node.
        """
        return self.oc_pruning_mask.get_mask()

    def compute_mask(self):
        """
        Computes the pruning mask by iteratively adding SIMD groups to unpruned state
        based on their importance and the target resource utilization.
        """
        # Iteratively unprune the graph while monitoring the memory footprint.
        current_memory = self.memory_calculator.get_pruned_graph_memory(masks=self.oc_pruning_mask.get_mask(),
                                                                        include_padded_channels=self.fqc.is_simd_padding)
        if current_memory > self.target_resource_utilization.weights_memory:
            Logger.critical(f"Insufficient memory for the target resource utilization: current memory {current_memory}, "
                            f"target memory {self.target_resource_utilization.weights_memory}.")

        # Greedily unprune groups (by setting their mask to 1) until the memory target is met
        # or all channels unpruned.
        while current_memory < self.target_resource_utilization.weights_memory and self.oc_pruning_mask.has_pruned_channel():
            # Select the best SIMD group (best means highest score which means most sensitive group)
            # to add based on the scores.
            node_to_remain, group_to_remain_idx = self._get_most_sensitive_simd_group_candidate()
            self.oc_pruning_mask.set_mask_value_for_simd_group(node=node_to_remain,
                                                               group_index=group_to_remain_idx,
                                                               mask_indicator=MaskIndicator.REMAINED)
            current_memory = self.memory_calculator.get_pruned_graph_memory(masks=self.oc_pruning_mask.get_mask(),
                                                                            include_padded_channels=self.fqc.is_simd_padding)

        # If the target memory is exceeded, revert the last addition.
        if current_memory > self.target_resource_utilization.weights_memory:
            self.oc_pruning_mask.set_mask_value_for_simd_group(node=node_to_remain,
                                                               group_index=group_to_remain_idx,
                                                               mask_indicator=MaskIndicator.PRUNED)



    def _get_most_sensitive_simd_group_candidate(self) -> Tuple[BaseNode, int]:
        """
        Identifies the most sensitive SIMD group for pruning based on the importance scores.

        Returns:
            Tuple[BaseNode, int]: The node and group index of the most sensitive SIMD group.
        """

        best_score = -np.inf
        best_node = None
        best_group_idx = -1

        for node, mask in self.oc_pruning_mask.get_mask_simd().items():
            # Get the index of the first zero in the mask. A zero indicates a prunable channel group.
            group_idx = int(np.argmax(mask == 0))

            # If group_idx is 0, it means there are no zeros in the mask, so this group is not prunable.
            if group_idx != 0:
                score = self.simd_groups_scores[node][group_idx]
                # If the score for this group is better than the best score found so far, update the best score.
                if score > best_score:
                    best_score = score
                    best_node = node
                    best_group_idx = group_idx

        if best_node is None:
            Logger.error("No prunable SIMD group found.")

        return best_node, best_group_idx

