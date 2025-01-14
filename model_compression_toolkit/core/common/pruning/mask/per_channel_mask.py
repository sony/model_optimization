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
from enum import Enum

import numpy as np
from typing import List, Dict, Tuple

from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core.common.pruning.memory_calculator import MemoryCalculator
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import PruningFrameworkImplementation
from model_compression_toolkit.logger import Logger

class MaskIndicator(Enum):
    """
    Enum class for indicating the status of channels in a pruning mask.

    PRUNED: Represents channels that are removed or pruned from the model.
    REMAINED: Represents channels that are kept or remain unpruned in the model.
    """
    PRUNED = 0
    REMAINED = 1



class PerChannelMask:
    def __init__(self, prunable_nodes: List[BaseNode], fw_info: FrameworkInfo):
        """
        Initializes the PerChannelMask with prunable nodes and framework information.
        This class is responsible for maintaining and updating the pruning masks for each
        prunable node in the model. The mask is an array indicating whether each output channel
        of a node is pruned (0) or remained (1).

        Args:
            prunable_nodes: List of nodes in the model that are subject to pruning.
            fw_info: Framework-specific information required for pruning operations.
        """
        self.prunable_nodes = prunable_nodes
        self.fw_info = fw_info
        self._mask = None  # Initialize the mask dictionary
        self._init_masks()  # Call to initialize masks for each prunable node

    def get_mask(self) -> Dict[BaseNode, np.ndarray]:
        """
        Retrieves the current pruning masks for all prunable nodes in the model.

        Returns:
            A dictionary mapping each prunable node to its corresponding pruning mask.
        """
        return self._mask

    def set_mask_value_for_simd_group(self, node: BaseNode, channel_idx: int, mask_indicator: MaskIndicator):
        """
        Sets the mask value for a specific channel of a prunable node.

        Args:
            node: The prunable node to update the mask for.
            channel_idx: The index of the channel to update in the mask.
            mask_indicator: The new value to set in the mask (either PRUNED or REMAINED).
        """
        if mask_indicator not in [MaskIndicator.PRUNED, MaskIndicator.REMAINED]:
            Logger.critical("Mask value must be either 'MaskIndicator.PRUNED' or 'MaskIndicator.REMAINED'")  # pragma: no cover
        self._mask[node][channel_idx] = mask_indicator.value

    def has_pruned_channel(self) -> bool:
        """
        Determines if there is at least one pruned channel across all nodes in the model.

        Returns:
            True if there is at least one pruned channel, False otherwise.
        """
        return any(MaskIndicator.PRUNED.value in mask for mask in self._mask.values())

    def _init_masks(self):
        """
        Initializes the pruning masks for each prunable node in the model.
        Sets the initial mask for each node as an array of zeros (indicating all channels are
        initially pruned).
        """
        self._mask = {}  # Initialize the dictionary for pruning masks.
        for prunable_node in self.prunable_nodes:
            num_oc = self._compute_num_of_out_channels(prunable_node)  # Number of output channels for the node.
            layer_mask = np.full(num_oc, MaskIndicator.PRUNED.value) # Initialize the mask with zeros.
            self._mask[prunable_node] = layer_mask

    def _compute_num_of_out_channels(self, node: BaseNode) -> int:
        """
        Computes the number of output channels for a given node.

        Args:
            node (BaseNode): The node whose output channels are to be counted.

        Returns:
            int: Number of output channels for the node.
        """
        kernel_attr = self.fw_info.get_kernel_op_attributes(node.type)[0]
        oc_axis = self.fw_info.kernel_channels_mapping.get(node.type)[0]
        num_oc = node.get_weights_by_keys(kernel_attr).shape[oc_axis]
        return num_oc

