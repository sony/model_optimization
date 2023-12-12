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
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities


class PerChannelMask:
    def __init__(self,
                 prunable_nodes: List[BaseNode],
                 fw_info: FrameworkInfo):
        """

        Args:
            prunable_nodes:
            fw_info:
        """
        self.prunable_nodes = prunable_nodes
        self.fw_info = fw_info
        self._mask = None  # Dictionary from a node to a mask at length of number of output channels it has (1-channel remains, 0-channel pruned)
        self._init_masks()


    def get_mask(self) -> Dict[BaseNode, np.ndarray]:
        return self._mask


    def set_mask_value_for_simd_group(self,
                                      node: BaseNode,
                                      channel_idx: int,
                                      value: int):

        assert value in [0, 1], "Mask value must be either 0 or 1."
        self._mask[node][channel_idx] = value


    def has_pruned_channel(self) -> bool:
        """
        Checks if there is at least one channel marked for pruning in any node mask.

        Returns:
            True if there is at least one channel to be pruned, False otherwise.
        """
        return any(0 in mask for mask in self._mask.values())


    def _init_masks(self):
        """
        Initializes the pruning masks for each prunable node in the graph.

        This method creates two types of masks:
        1. `self.mask`: A detailed mask for each output channel of the prunable nodes.
        """
        self._mask = {}  # Dictionary to store the fine-grained mask for each prunable node.

        # Iterate over all prunable nodes to initialize their masks.
        for prunable_node in self.prunable_nodes:
            # Determine the number of output channels for the node.
            num_oc = prunable_node.get_weights_by_keys(self.fw_info.get_kernel_op_attributes(prunable_node.type)[0]).shape[self.fw_info.kernel_channels_mapping.get(prunable_node.type)[0]]
            # Initialize a mask with zeros for each output channel.
            layer_mask = np.zeros(num_oc)
            self._mask[prunable_node] = layer_mask

