import numpy as np
from typing import List, Dict

from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSection, PruningSectionMask


class MemoryCalculator:
    """
    MemoryCalculator is a class that computes the memory usage of a pruned graph.
    It takes into account the pruning masks applied to each node and computes the memory
    accordingly.
    """

    def __init__(self,
                 graph: Graph,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation):
        """
        Initialize the MemoryCalculator.

        Args:
            graph: The computational graph of the model.
            fw_info: Framework-specific information and utilities.
            fw_impl: Framework-specific implementation details.
        """
        self.graph = graph
        self.fw_info = fw_info
        self.fw_impl = fw_impl

    def get_pruned_graph_memory(self,
                                masks: Dict[BaseNode, np.ndarray],
                                fw_impl,
                                include_null_channels: bool = True) -> float:

        # Total number of parameters after pruning
        total_nparams = 0.0

        # Retrieve all the pruning sections in the graph
        pruning_sections = self.graph.get_pruning_sections(self.fw_info, fw_impl)

        # Calculate the number of parameters for nodes that are not pruned
        total_nparams += self.get_nparams_of_nonpruned_nodes(pruning_sections, include_null_channels)

        # Calculate the number of parameters for nodes within pruning sections
        total_nparams += self.get_nparams_of_pruning_sections(masks, pruning_sections, include_null_channels)

        # Subtract the number of parameters for nodes that are shared between pruning sections
        total_nparams -= self.get_nparams_of_shared_nodes(masks, pruning_sections, include_null_channels)

        # Multiply by the size of the data type to get the memory in bytes
        return total_nparams * 4.

    def get_nparams_of_shared_nodes(self,
                                    masks: Dict[BaseNode, np.ndarray],
                                    pruning_sections: List[PruningSection],
                                    include_null_channels) -> int:

        nparams = 0
        # Identify nodes that are at the end of one section and the start of another
        shared_nodes = self._get_nodes_from_adjacent_sections(pruning_sections)
        for node in shared_nodes:
            # Get the input mask for the node if it exists
            node_input_mask = self._get_node_input_mask(node, pruning_sections, masks)
            # Get the output mask for the node if it exists
            node_output_mask = masks.get(node)
            # Calculate the number of remaining parameters for the shared node after pruning
            nparams += self.fw_impl.get_pruned_node_num_params(node,
                                                               node_input_mask,
                                                               node_output_mask,
                                                               self.fw_info,
                                                               include_null_channels)
        return nparams

    def get_nparams_of_pruning_sections(self, masks, pruning_sections, include_null_channels:bool):
        """

        Args:
            masks:
            pruning_sections:

        Returns:

        """
        nparams = 0
        for pruning_section in pruning_sections:
            pruning_section_mask = self.get_section_mask_from_node_mask(masks, pruning_section, pruning_sections)
            nparams += self._get_pruning_section_num_params(pruning_section, pruning_section_mask, include_null_channels)
        return nparams

    def get_section_mask_from_node_mask(self, masks, pruning_section, pruning_sections):
        """

        Args:
            masks:
            pruning_section:
            pruning_sections:

        Returns:

        """
        # Determine masks for input channels of the first node and output channels of the second node.
        first_node_input_channels_mask = self._get_node_input_mask(pruning_section.entry_node,
                                                                   pruning_sections,
                                                                   masks)
        second_node_output_mask = masks.get(pruning_section.exit_node)

        # Create the pruning section mask.
        pruning_section_mask = PruningSectionMask(entry_input_mask=first_node_input_channels_mask,
                                                  entry_output_mask=masks.get(pruning_section.entry_node),
                                                  exit_input_mask=masks.get(pruning_section.entry_node),
                                                  exit_output_mask=second_node_output_mask)

        return pruning_section_mask

    def get_nparams_of_nonpruned_nodes(self, pruning_sections, include_null_channels):
        """

        Args:
            pruning_sections:

        Returns:

        """
        total_nparams = 0
        # Collect all nodes to prune from the pruning sections.
        nodes_to_prune = set([node for section in pruning_sections for node in section.get_all_nodes()])
        # Calculate the num of params for non-prunable nodes.
        for n in self.graph.nodes:
            if n not in nodes_to_prune:
                node_nparams = sum(n.get_num_parameters(self.fw_info))
                if include_null_channels:
                    num_oc = n.output_shape[-1]
                    nparams_per_oc = node_nparams/num_oc
                    num_oc_include_null_channels = np.ceil(num_oc/n.get_simd())*n.get_simd()
                    node_nparams = num_oc_include_null_channels*nparams_per_oc
                total_nparams += node_nparams
        return total_nparams

    def _get_node_input_mask(self,
                             node: BaseNode, pruning_sections: List[PruningSection],
                             masks: Dict[BaseNode, np.ndarray]) -> np.ndarray:
        """

        Args:
            node:
            pruning_sections:
            masks:

        Returns:

        """
        for section in pruning_sections:
            if node == section.exit_node:
                return masks.get(section.entry_node)
        return None

    def _get_nodes_from_adjacent_sections(self, pruning_sections: List[PruningSection]) -> List[BaseNode]:
        """

        Args:
            pruning_sections:

        Returns:

        """
        input_nodes = set(section.entry_node for section in pruning_sections)
        output_nodes = set(section.exit_node for section in pruning_sections)
        return list(input_nodes.intersection(output_nodes))

    def _get_pruning_section_num_params(self,
                                        pruning_section: PruningSection,
                                        pruning_section_mask: PruningSectionMask,
                                        include_null_channels: bool) -> int:
        """

        Args:
            pruning_section:
            pruning_section_mask:

        Returns:

        """
        # Number of params for the first node in the section.
        first_node_nparams = self.fw_impl.get_pruned_node_num_params(
            pruning_section.entry_node,
            pruning_section_mask.entry_input_mask,
            pruning_section_mask.entry_output_mask,
            self.fw_info,
            include_null_channels)

        # Sum number of params for all intermediate nodes in the section.
        total_inter_nodes_nparams = sum(
            self.fw_impl.get_pruned_node_num_params(
                inter_node,
                pruning_section_mask.entry_output_mask,
                pruning_section_mask.entry_output_mask,
                self.fw_info,
                include_null_channels) for inter_node in pruning_section.intermediate_nodes)

        # Number of params for the last node in the section.
        second_node_nparams = self.fw_impl.get_pruned_node_num_params(
            pruning_section.exit_node,
            pruning_section_mask.exit_input_mask,
            pruning_section_mask.exit_output_mask,
            self.fw_info,
            include_null_channels)

        return first_node_nparams + total_inter_nodes_nparams + second_node_nparams
