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
from model_compression_toolkit.core.common.pruning.mask.per_channel_mask import PerChannelMask, MaskIndicator
from model_compression_toolkit.core.common.pruning.memory_calculator import MemoryCalculator
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import PruningFrameworkImplementation
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities


class PerSIMDGroupMask:
    def __init__(self,
                 prunable_nodes: List[BaseNode],
                 fw_info: FrameworkInfo,
                 simd_groups_indices: Dict[BaseNode, List[List[int]]]):
        """

        Args:
            prunable_nodes:
            fw_info:
            simd_groups_indices:
        """
        self.per_channel_mask = PerChannelMask(prunable_nodes=prunable_nodes,
                                               fw_info=fw_info)
        self.prunable_nodes = prunable_nodes
        self.fw_info = fw_info
        self.simd_groups_indices = simd_groups_indices

        self._mask_simd = None # Dictionary from a node to a mask at length of number of SIMD groups it has (1-group remains, 0-group pruned)

        # Create the masks and set the first SIMD group of channels to be unpruned.
        self._init_masks()
        self._update_mandatory_mask()

    def get_mask_simd(self) -> Dict[BaseNode, np.ndarray]:
        return self._mask_simd

    def get_mask(self) -> Dict[BaseNode, np.ndarray]:
        return self.per_channel_mask.get_mask()


    def set_mask_value_for_simd_group(self,
                                      node: BaseNode,
                                      group_index: int,
                                      mask_indicator: MaskIndicator):
        """
        Updates the mask for a specific SIMD group of a node.

        Args:
            node: The prunable node for which the mask is to be updated.
            group_index: Index of the SIMD group within the node.
            mask_indicator: The new value (0 or 1) to set for the group mask.
        """
        assert mask_indicator in [MaskIndicator.PRUNED,
                                  MaskIndicator.REMAINED], ("Mask value must be either MaskIndicator.PRUNED "
                                                            "or MaskIndicator.REMAINED.")
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
        Initializes the pruning masks for each prunable node in the graph.
        `self._mask_simd`: A mask at the level of SIMD groups for each prunable node.
        """
        self._mask_simd = {}  # Dictionary to store the SIMD group level mask for each prunable node.

        # Iterate over all prunable nodes to initialize their masks.
        for prunable_node in self.prunable_nodes:
            num_oc = len(self.per_channel_mask.get_mask()[prunable_node])
            # Calculate the number of SIMD groups in the layer.
            layer_num_simd_groups = int(max(np.ceil(num_oc / prunable_node.get_simd()), 1))
            # Initialize a SIMD group level mask with zeros.
            layer_mask_per_simd_group = np.zeros(layer_num_simd_groups)
            # Store the initialized masks in their respective dictionaries.
            self._mask_simd[prunable_node] = layer_mask_per_simd_group

    def _update_mandatory_mask(self):
        for prunable_node in self.prunable_nodes:
            # Mark the first SIMD group of channels as mandatory (unpruned) for each prunable node.
            # This is done by setting the first group's value in the SIMD mask to 1 (unpruned).
            self.set_mask_value_for_simd_group(node=prunable_node,
                                               group_index=0,
                                               mask_indicator=MaskIndicator.REMAINED)


