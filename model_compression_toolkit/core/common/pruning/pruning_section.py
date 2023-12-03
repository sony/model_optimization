from typing import List

import numpy as np

from model_compression_toolkit.core.common.graph.base_node import BaseNode

class PruningSectionMask:
    """
    Represents the masks to be applied to a pruning section of a neural network.
    This includes masks for both input and output channels at the entry and exit nodes of the section.

    Attributes:
        entry_input_mask (np.ndarray): Mask for input channels of the entry node.
        entry_output_mask (np.ndarray): Mask for output channels of the entry node.
        exit_input_mask (np.ndarray): Mask for input channels of the exit node.
        exit_output_mask (np.ndarray): Mask for output channels of the exit node.
    """

    def __init__(self,
                 entry_input_mask: np.ndarray, # TODO:entry_node_ic_mask
                 entry_output_mask: np.ndarray,
                 exit_input_mask: np.ndarray,
                 exit_output_mask: np.ndarray):
        self.entry_input_mask = entry_input_mask
        self.entry_output_mask = entry_output_mask
        self.exit_input_mask = exit_input_mask
        self.exit_output_mask = exit_output_mask


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

    def get_all_nodes(self) -> List[BaseNode]:
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
                                 fw_impl,
                                 fw_info):
        """
        Apply the provided pruning section mask to all nodes within the pruning section.

        Args:
            pruning_section_mask (PruningSectionMask): The mask to be applied to the pruning section.
            fw_impl: Framework-specific implementation for applying the mask.
            fw_info: Framework-specific information needed to apply the mask.
        """
        fw_impl.prune_entry_node(node=self.entry_node,
                                 output_mask=pruning_section_mask.entry_output_mask,
                                 fw_info=fw_info)

        for inter_node in self.intermediate_nodes:
            fw_impl.prune_intermediate_node(node=inter_node,
                                            input_mask=pruning_section_mask.entry_output_mask,
                                            output_mask=pruning_section_mask.entry_output_mask,
                                            fw_info=fw_info)

        fw_impl.prune_exit_node(self.exit_node,
                                input_mask=pruning_section_mask.exit_input_mask,
                                fw_info=fw_info)

