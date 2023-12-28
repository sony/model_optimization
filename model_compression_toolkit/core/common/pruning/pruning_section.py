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

from typing import List, Any

import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_node import BaseNode


class PruningSectionMask:
    """
    Represents the masks to be applied to a pruning section of a neural network.
    This includes masks for both input and output channels at the entry and exit nodes of the section.

    Attributes:
        entry_node_ic_mask (np.ndarray): Mask for input channels of the entry node.
        entry_node_oc_mask (np.ndarray): Mask for output channels of the entry node.
        exit_node_ic_mask (np.ndarray): Mask for input channels of the exit node.
        exit_node_oc_mask (np.ndarray): Mask for output channels of the exit node.
    """

    def __init__(self,
                 entry_node_ic_mask: np.ndarray = None,
                 entry_node_oc_mask: np.ndarray = None,
                 exit_node_ic_mask: np.ndarray = None,
                 exit_node_oc_mask: np.ndarray = None):
        self.entry_node_ic_mask = entry_node_ic_mask
        self.entry_node_oc_mask = entry_node_oc_mask
        self.exit_node_ic_mask = exit_node_ic_mask
        self.exit_node_oc_mask = exit_node_oc_mask


class PruningSection:
    """
    Represents a section in a graph to be pruned, consisting of an entry node,
    intermediate nodes, and an exit node.

    Attributes:
        entry_node (BaseNode): The first node in the pruning section.
        intermediate_nodes (List[BaseNode]): List of nodes between the entry and exit nodes.
        exit_node (BaseNode): The last node in the pruning section.
    """

    def __init__(self,
                 entry_node: BaseNode,
                 intermediate_nodes: List[BaseNode],
                 exit_node: BaseNode):
        self.entry_node = entry_node
        self.intermediate_nodes = intermediate_nodes
        self.exit_node = exit_node

    def get_all_section_nodes(self) -> List[BaseNode]:
        """
        Returns a list of all nodes in the pruning section, including the entry,
         intermediate, and exit nodes.

        Returns:
            List[BaseNode]: List of all nodes in the pruning section.
        """
        nodes = [self.entry_node] + self.intermediate_nodes + [self.exit_node]
        return nodes

    def apply_inner_section_mask(self,
                                 pruning_section_mask: PruningSectionMask,
                                 fw_impl: Any,
                                 fw_info: FrameworkInfo):
        """
        Apply the provided pruning section mask to all nodes within the pruning section.

        Args:
            pruning_section_mask (PruningSectionMask): The mask to be applied to the pruning section.
            fw_impl (PruningFrameworkImplementation): Framework-specific implementation for applying the mask.
            fw_info (FrameworkInfo): Framework-specific information needed to apply the mask.
        """
        fw_impl.prune_entry_node(node=self.entry_node,
                                 output_mask=pruning_section_mask.entry_node_oc_mask,
                                 fw_info=fw_info)

        for inter_node in self.intermediate_nodes:
            fw_impl.prune_intermediate_node(node=inter_node,
                                            input_mask=pruning_section_mask.entry_node_oc_mask,
                                            output_mask=pruning_section_mask.entry_node_oc_mask,
                                            fw_info=fw_info)

        fw_impl.prune_exit_node(self.exit_node,
                                input_mask=pruning_section_mask.exit_node_ic_mask,
                                fw_info=fw_info)

    @staticmethod
    def has_matching_channel_count(exit_node: BaseNode,
                                   corresponding_entry_node: BaseNode,
                                   fw_info: FrameworkInfo) -> bool:
        """
        Checks if the number of input channels of the exit node matches the number of output channels
        of its corresponding entry node.

        Args:
            exit_node (BaseNode): The node exiting a pruning section.
            corresponding_entry_node (BaseNode): The entry node of the subsequent pruning section.

        Returns:
            bool: True if the channel counts match, False otherwise.
        """
        _, exit_input_channel_axis = fw_info.kernel_channels_mapping.get(exit_node.type)
        entry_output_channel_axis, _ = fw_info.kernel_channels_mapping.get(corresponding_entry_node.type)

        exit_node_attr = fw_info.get_kernel_op_attributes(exit_node.type)[0]
        entry_node_attr = fw_info.get_kernel_op_attributes(corresponding_entry_node.type)[0]

        exit_input_channels = exit_node.get_weights_by_keys(exit_node_attr).shape[exit_input_channel_axis]
        entry_output_channels = corresponding_entry_node.get_weights_by_keys(entry_node_attr).shape[entry_output_channel_axis]

        return exit_input_channels == entry_output_channels
