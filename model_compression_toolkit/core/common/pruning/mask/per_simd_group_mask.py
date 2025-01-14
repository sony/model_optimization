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
from model_compression_toolkit.core.common.pruning.mask.per_channel_mask import PerChannelMask, MaskIndicator
from model_compression_toolkit.core.common.pruning.memory_calculator import MemoryCalculator
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import PruningFrameworkImplementation
from model_compression_toolkit.logger import Logger

class PerSIMDGroupMask:
    def __init__(self,
                 prunable_nodes: List[BaseNode],
                 fw_info: FrameworkInfo,
                 simd_groups_indices: Dict[BaseNode, List[List[int]]]):
        """
        Initializes a mask calculator for SIMD groups in prunable nodes.
        Manages both per-channel and per-SIMD-group masks.

        Args:
            prunable_nodes: List of nodes that can be pruned.
            fw_info: Framework-specific information.
            simd_groups_indices: A dictionary mapping each node to its SIMD groups' indices.
        """
        # Initialize the per-channel mask
        self.per_channel_mask = PerChannelMask(prunable_nodes=prunable_nodes, fw_info=fw_info)
        self.prunable_nodes = prunable_nodes
        self.fw_info = fw_info
        self.simd_groups_indices = simd_groups_indices
        self._mask_simd = None  # Initialize the SIMD group mask dictionary
        self._init_masks()  # Initialize masks for each prunable node
        self._update_mandatory_mask()  # Ensure at least one SIMD group remains unpruned

    def get_mask_simd(self) -> Dict[BaseNode, np.ndarray]:
        """
        Retrieves the current SIMD group masks for all prunable nodes.

        Returns:
            A dictionary mapping each prunable node to its corresponding SIMD group mask.
        """
        return self._mask_simd

    def get_mask(self) -> Dict[BaseNode, np.ndarray]:
        """
        Retrieves the current per-channel masks for all prunable nodes.

        Returns:
            A dictionary mapping each prunable node to its corresponding per-channel mask.
        """
        return self.per_channel_mask.get_mask()

    def set_mask_value_for_simd_group(self,
                                      node: BaseNode,
                                      group_index: int,
                                      mask_indicator: MaskIndicator):
        """
        Sets the mask value for a specific SIMD group of a prunable node.

        Args:
            node: The prunable node to update the mask for.
            group_index: The index of the SIMD group to update in the mask.
            mask_indicator: The new value to set in the mask (either PRUNED or REMAINED).
        """
        if mask_indicator not in [MaskIndicator.PRUNED, MaskIndicator.REMAINED]:
            Logger.critical("Mask value must be either 'MaskIndicator.PRUNED' or 'MaskIndicator.REMAINED'")  # pragma: no cover

        # Update the SIMD group mask and corresponding per-channel mask
        self._mask_simd[node][group_index] = mask_indicator.value
        node_mask_indices = self.simd_groups_indices[node][group_index]
        for idx in node_mask_indices:
            self.per_channel_mask.set_mask_value_for_simd_group(node=node,
                                                                channel_idx=idx,
                                                                mask_indicator=mask_indicator)
    def has_pruned_channel(self) -> bool:
        """
        Checks if there is at least one channel marked for pruning in any node mask.

        Returns:
            True if there is at least one channel to be pruned, False otherwise.
        """
        return self.per_channel_mask.has_pruned_channel()

    def _init_masks(self):
        """
        Initializes the SIMD group masks for each prunable node.
        Sets the initial mask for each node as an array of zeros (indicating
        all groups are initially pruned).
        """
        self._mask_simd = {}  # Initialize the dictionary for SIMD group masks.
        for prunable_node in self.prunable_nodes:
            num_groups = len(self.simd_groups_indices[prunable_node])  # Number of SIMD groups for the node.
            layer_mask_per_simd_group = np.full(num_groups, MaskIndicator.PRUNED.value)  # Initialize the mask with zeros.
            self._mask_simd[prunable_node] = layer_mask_per_simd_group

    def _update_mandatory_mask(self):
        """
        Updates the mandatory masks for each prunable node to ensure at least one SIMD
        group remains unpruned.
        """
        for prunable_node in self.prunable_nodes:
            # Mark the first SIMD group as mandatory (unpruned).
            self.set_mask_value_for_simd_group(node=prunable_node,
                                               group_index=0,
                                               mask_indicator=MaskIndicator.REMAINED)

