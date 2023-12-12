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
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.pruning.memory_calculator import MemoryCalculator
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import PruningFrameworkImplementation
from model_compression_toolkit.core.common.pruning.mask.per_simd_group_mask import PerSIMDGroupMask
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities


class GreedyMaskCalculator:
    def __init__(self,
                 prunable_nodes: List[BaseNode],
                 fw_info: FrameworkInfo,
                 score_by_node: Dict[BaseNode, np.ndarray],
                 target_kpi: KPI,
                 graph: Graph,
                 fw_impl: PruningFrameworkImplementation,
                 tpc: TargetPlatformCapabilities,
                 simd_groups_indices: Dict[BaseNode, List[List[int]]]):
        """

        Args:
            prunable_nodes:
            fw_info:
            score_by_node:
            target_kpi:
            graph:
            fw_impl:
            tpc:
            simd_groups_indices:
        """
        self.prunable_nodes = prunable_nodes
        self.fw_info = fw_info
        self.target_kpi = target_kpi
        self.graph = graph
        self.fw_impl = fw_impl
        self.tpc = tpc

        self.simd_groups_indices = simd_groups_indices
        self.simd_groups_scores = score_by_node

        self.oc_pruning_mask = PerSIMDGroupMask(prunable_nodes=prunable_nodes,
                                                fw_info=fw_info,
                                                simd_groups_indices=simd_groups_indices)

        self.memory_calculator = MemoryCalculator(graph=graph,
                                                  fw_info=fw_info,
                                                  fw_impl=fw_impl)


    def get_mask(self) -> Dict[BaseNode, np.ndarray]:
        return self.oc_pruning_mask.get_mask()

    def compute_mask(self):
        # Iteratively prune the graph while monitoring the memory footprint.
        current_memory = self.memory_calculator.get_pruned_graph_memory(masks=self.oc_pruning_mask.get_mask(),
                                                                        include_padded_channels=self.tpc.is_simd_padding())
        if current_memory > self.target_kpi.weights_memory:
            Logger.error(f"Minimal required memory is {current_memory}, "
                         f"but target KPI is {self.target_kpi.weights_memory}")

        # Greedily add groups to the mask until the memory target is met or all channels remains.
        while current_memory < self.target_kpi.weights_memory and self.oc_pruning_mask.has_pruned_channel():
            # Select the best SIMD group to add based on the scores.
            node_to_remain, group_to_remain_idx = self._get_best_simd_group_candidate()
            self.oc_pruning_mask.set_mask_value_for_simd_group(node=node_to_remain,
                                                               group_index=group_to_remain_idx,
                                                               value=1)
            current_memory = self.memory_calculator.get_pruned_graph_memory(masks=self.oc_pruning_mask.get_mask(),
                                                                            include_padded_channels=self.tpc.is_simd_padding())

        # If the target memory is exceeded, revert the last addition.
        if current_memory > self.target_kpi.weights_memory:
            self.oc_pruning_mask.set_mask_value_for_simd_group(node=node_to_remain,
                                                               group_index=group_to_remain_idx,
                                                               value=0)



    def _get_best_simd_group_candidate(self) -> Tuple[BaseNode, int]:
        """
        Finds the best SIMD group candidate for pruning.

        Returns:
            A tuple containing the node with the best SIMD group and the group index.
        """
        # Initialize variables to track the best score and corresponding node and group index.
        best_score = -np.inf
        best_node = None
        best_group_idx = -1

        for node, mask in self.oc_pruning_mask.get_mask_simd().items():
            # Get the index of the first zero in the mask. A zero indicates a prunable channel group.
            group_idx = int(np.argmax(mask == 0))

            # If group_idx is 0, it means there are no zeros in the mask, so this group is not prunable.
            if group_idx != 0:
                score = np.sum(self.simd_groups_scores[node][group_idx])
                # If the score for this group is better than the best score found so far, update the best score.
                if score > best_score:
                    best_score = score
                    best_node = node
                    best_group_idx = group_idx

        if best_node is None:
            raise ValueError("No prunable SIMD group found.")

        return best_node, best_group_idx

