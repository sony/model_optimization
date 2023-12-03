import numpy as np
from typing import List, Dict

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import \
    PruningFrameworkImplementation
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
                 fw_impl: PruningFrameworkImplementation):
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
                                include_padded_channels: bool) -> float:
        nparams = self.get_pruned_graph_num_params(masks=masks,
                                                   include_padded_channels=include_padded_channels)
        return nparams * 4.

    def get_pruned_graph_num_params(self,
                                    masks: Dict[BaseNode, np.ndarray],
                                    include_padded_channels: bool) -> int:

        # Total number of parameters after pruning
        total_nparams = 0

        # Retrieve all the pruning sections in the graph
        pruning_sections = self.graph.get_pruning_sections(self.fw_info, self.fw_impl)

        # Calculate the number of parameters for nodes that are not pruned
        total_nparams += self.get_nparams_of_nonpruned_nodes(pruning_sections, include_padded_channels)

        # Calculate the number of parameters for nodes within pruning sections
        total_nparams += self.get_nparams_of_pruning_sections(masks, pruning_sections, include_padded_channels)

        # Subtract the number of parameters for nodes that are shared between pruning sections
        total_nparams -= self.get_nparams_of_shared_nodes(masks, pruning_sections, include_padded_channels)

        return total_nparams

    def get_nparams_of_shared_nodes(self,
                                    masks: Dict[BaseNode, np.ndarray],
                                    pruning_sections: List[PruningSection],
                                    include_padded_channels) -> int:

        nparams = 0
        # Identify nodes that are at the end of one section and the start of another
        shared_nodes = self._get_nodes_from_adjacent_sections(pruning_sections)
        for node in shared_nodes:
            # Get the input mask for the node if it exists
            node_input_mask = self._get_node_input_mask(node, pruning_sections, masks)
            # Get the output mask for the node if it exists
            node_output_mask = masks.get(node)
            # Calculate the number of remaining parameters for the shared node after pruning
            nparams += self.get_pruned_node_num_params(node,
                                                       node_input_mask,
                                                       node_output_mask,
                                                       self.fw_info,
                                                       include_padded_channels)
        return nparams

    def get_nparams_of_pruning_sections(self, masks, pruning_sections, include_padded_channels:bool):
        """

        Args:
            masks:
            pruning_sections:

        Returns:

        """
        nparams = 0
        for pruning_section in pruning_sections:
            pruning_section_mask = self.get_section_mask_from_node_mask(masks,
                                                                        pruning_section,
                                                                        pruning_sections)
            nparams += self._get_pruning_section_num_params(pruning_section,
                                                            pruning_section_mask,
                                                            include_padded_channels)
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
        pruning_section_mask = PruningSectionMask(entry_node_ic_mask=first_node_input_channels_mask,
                                                  entry_node_oc_mask=masks.get(pruning_section.entry_node),
                                                  exit_node_ic_mask=masks.get(pruning_section.entry_node),
                                                  exit_node_oc_mask=second_node_output_mask)

        return pruning_section_mask

    def get_nparams_of_nonpruned_nodes(self, pruning_sections, include_padded_channels:bool):

        total_nparams = 0
        # Collect all nodes to prune from the pruning sections.
        nodes_to_prune = set([node for section in pruning_sections for node in section.get_all_nodes()])
        # Calculate the num of params for non-prunable nodes.
        for n in self.graph.nodes:
            if n not in nodes_to_prune:
                node_nparams = sum(n.get_num_parameters(self.fw_info))
                if include_padded_channels:
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
                                        include_padded_channels: bool) -> int:

        # Number of params for the first node in the section.
        first_node_nparams = self.get_pruned_node_num_params(
            pruning_section.entry_node,
            pruning_section_mask.entry_node_ic_mask,
            pruning_section_mask.entry_node_oc_mask,
            self.fw_info,
            include_padded_channels)

        # Sum number of params for all intermediate nodes in the section.
        total_inter_nodes_nparams = sum(
            self.get_pruned_node_num_params(
                inter_node,
                pruning_section_mask.entry_node_oc_mask,
                pruning_section_mask.entry_node_oc_mask,
                self.fw_info,
                include_padded_channels) for inter_node in pruning_section.intermediate_nodes)

        # Number of params for the last node in the section.
        second_node_nparams = self.get_pruned_node_num_params(
            pruning_section.exit_node,
            pruning_section_mask.exit_node_ic_mask,
            pruning_section_mask.exit_node_oc_mask,
            self.fw_info,
            include_padded_channels)

        return first_node_nparams + total_inter_nodes_nparams + second_node_nparams


    def get_pruned_node_num_params(self,
                                         node: BaseNode,
                                         input_mask: np.ndarray,
                                         output_mask: np.ndarray,
                                         fw_info: FrameworkInfo,
                                         include_padded_channels: bool):
        """
        Calculates the number of parameters in a pruned node of a Keras model.

        Args:
            node: The node whose parameters are to be counted.
            input_mask: Mask to be applied to the input channels.
            output_mask: Mask to be applied to the output channels.
            fw_info: Framework-specific information object.
            include_padded_channels: Boolean flag to include or exclude null channels in the count.

        Returns:
            Integer representing the number of parameters in the pruned node.
        """

        total_params = 0
        if fw_info.is_kernel_op(node.type):
            # Obtain axes info for kernel operations.
            oc_axis, ic_axis = fw_info.kernel_channels_mapping.get(node.type)
            kernel_attr = fw_info.get_kernel_op_attributes(node.type)[0]
            for w_attr, w in node.weights.items():
                # Check if the weight attribute is the kernel.
                if kernel_attr in w_attr:
                    # Handle input and output masks, ensuring they are boolean arrays.
                    input_mask = np.ones(w.shape[ic_axis], dtype=bool) if input_mask is None else input_mask.astype(bool)
                    output_mask = np.ones(w.shape[oc_axis], dtype=bool) if output_mask is None else output_mask.astype(bool)

                    # Assert the input and output masks match the kernel dimensions.
                    assert w.shape[ic_axis] == len(input_mask), (f"Kernel num of input channels: {w.shape[ic_axis]}, but mask len is {len(input_mask)} for node {node}")
                    assert w.shape[oc_axis] == len(output_mask), (f"Kernel num of output channels: {w.shape[oc_axis]}, but mask len is {len(output_mask)} for node {node}")

                    # Apply masks to the kernel and calculate the remaining parameters.
                    pruned_w = np.take(w, np.where(input_mask)[0], axis=ic_axis)
                    pruned_w = np.take(pruned_w, np.where(output_mask)[0], axis=oc_axis)
                    total_params += len(pruned_w.flatten())
                else:
                    # For non-kernel weights, apply the output mask only.
                    total_params += len(np.take(w, np.where(output_mask)[0]))

        else:
            # For non-kernel operations, apply the output mask to the last axis.
            # This part assumes that for non-kernel ops, all weights output channel axis is -1.
            for w_attr, w in node.weights.items():
                pruned_w = np.take(w, np.where(output_mask)[0], axis=-1) # TODO: get axis from fw-specific function
                total_params += pruned_w.size

        if include_padded_channels: # TODO: remove duplicate
            node_simd = node.get_simd()
            nparams_per_oc = total_params / np.sum(output_mask)
            num_oc_with_null_channels = np.ceil(np.sum(output_mask) / node_simd) * node_simd
            total_params = num_oc_with_null_channels * nparams_per_oc

        return total_params