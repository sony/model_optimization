import numpy as np
from typing import List, Dict

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import \
    PruningFrameworkImplementation
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSection, PruningSectionMask
from model_compression_toolkit.logger import Logger


class MemoryCalculator:


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
        """

        Args:
            masks:
            include_padded_channels:

        Returns:

        """
        nparams = self.get_pruned_graph_num_params(masks=masks,
                                                   include_padded_channels=include_padded_channels)
        return nparams * 4.

    def get_pruned_graph_num_params(self,
                                    masks: Dict[BaseNode, np.ndarray],
                                    include_padded_channels: bool) -> int:
        """

        Args:
            masks:
            include_padded_channels:

        Returns:

        """

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
        """

        Args:
            masks:
            pruning_sections:
            include_padded_channels:

        Returns:

        """

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
                                                       include_padded_channels)
        return nparams

    def get_nparams_of_pruning_sections(self,
                                        masks,
                                        pruning_sections,
                                        include_padded_channels: bool):
        """

        Args:
            masks:
            pruning_sections:
            include_padded_channels:

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

    def get_section_mask_from_node_mask(self,
                                        masks,
                                        pruning_section,
                                        pruning_sections):

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

    def get_nparams_of_nonpruned_nodes(self,
                                       pruning_sections,
                                       include_padded_channels: bool):
        """

        Args:
            pruning_sections:
            include_padded_channels:

        Returns:

        """

        total_nparams = 0
        # Collect all nodes to prune from the pruning sections.
        nodes_to_prune = set([node for section in pruning_sections for node in section.get_all_nodes()])
        # Calculate the num of params for non-prunable nodes.
        for n in self.graph.nodes:
            if n not in nodes_to_prune:
                # node_nparams = sum(n.get_num_parameters(self.fw_info))
                node_nparams = self.get_pruned_node_num_params(node=n,
                                                               input_mask=None,
                                                               output_mask=None,
                                                               include_padded_channels=include_padded_channels)
                # if include_padded_channels:
                #     node_nparams = self.get_node_nparams_with_padded_channels(node_nparams=node_nparams,
                #                                                               num_oc=n.output_shape[-1],
                #                                                               node_simd=n.get_simd())
                total_nparams += node_nparams
        return total_nparams

    def _get_node_input_mask(self,
                             node: BaseNode, pruning_sections: List[PruningSection],
                             masks: Dict[BaseNode, np.ndarray]) -> np.ndarray:

        for section in pruning_sections:
            if node == section.exit_node:
                return masks.get(section.entry_node)
        return None

    def _get_nodes_from_adjacent_sections(self,
                                          pruning_sections: List[PruningSection]) -> List[BaseNode]:

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
            include_padded_channels)

        # Sum number of params for all intermediate nodes in the section.
        total_inter_nodes_nparams = sum(
            self.get_pruned_node_num_params(
                inter_node,
                pruning_section_mask.entry_node_oc_mask,
                pruning_section_mask.entry_node_oc_mask,
                include_padded_channels) for inter_node in pruning_section.intermediate_nodes)

        # Number of params for the last node in the section.
        second_node_nparams = self.get_pruned_node_num_params(
            pruning_section.exit_node,
            pruning_section_mask.exit_node_ic_mask,
            pruning_section_mask.exit_node_oc_mask,
            include_padded_channels)

        return first_node_nparams + total_inter_nodes_nparams + second_node_nparams

    def get_pruned_node_num_params(self,
                                   node: BaseNode,
                                   input_mask: np.ndarray,
                                   output_mask: np.ndarray,
                                   include_padded_channels: bool):
        """
        Calculates the number of parameters in a pruned node of a model.

        Args:
            node: The node whose parameters are to be counted.
            input_mask: Mask to be applied to the input channels.
            output_mask: Mask to be applied to the output channels.
            include_padded_channels: Boolean flag to include or exclude padded channels (due to SIMD) in the count.

        Returns:
            Integer representing the number of parameters in the pruned node.
        """

        def _prune(w, mask, axis):
            mask = np.ones(w.shape[axis], dtype=bool) if mask is None else mask.astype(bool)
            assert w.shape[axis] == len(mask), (
                f"Kernel num of input channels: {w.shape[axis]}, but mask len is {len(mask)} for node {node}")
            pruned_w = np.take(w, np.where(mask)[0], axis=axis)
            return pruned_w

        total_params = 0
        attributes_and_oc_axis = self.fw_impl.get_node_attributes_with_io_axis(node, self.fw_info)
        for w_attr, w in node.weights.items():
            io_axis = [io_axis for attr, io_axis in attributes_and_oc_axis.items() if attr in w_attr]
            assert len(io_axis) == 1
            out_axis, in_axis = io_axis[0]
            if in_axis is not None and input_mask is not None:
                w = _prune(w, input_mask, in_axis)
            if out_axis is not None and output_mask is not None:
                w = _prune(w, output_mask, out_axis)
            total_params += w.size

        num_oc = np.sum(output_mask) if output_mask is not None else node.output_shape[-1]
        if include_padded_channels:
            total_params = self.get_node_nparams_with_padded_channels(node=node,
                                                                      node_nparams=total_params,
                                                                      num_oc=num_oc,
                                                                      node_simd=node.get_simd())

        return total_params

    def get_node_nparams_with_padded_channels(self,
                                              node: BaseNode,
                                              node_nparams: int,
                                              num_oc: int,
                                              node_simd: int):
        """

        Args:
            node_nparams:
            num_oc:
            node_simd:

        Returns:

        """
        nparams_per_oc = node_nparams / num_oc

        """
        Usually every layer has some number of params in each weight tensor dedicated for a single output-channel. 
        Sometimes not. For example: Keras Normalize layer with 3 output channels has 3 weights where 2 of them are 
        tensors of length 3 and a single scalar that is used for all 3 output channels.
        """
        if int(nparams_per_oc)!=nparams_per_oc:
            Logger.warning(
                f" Found a node {node.name} with weights that are not uniformly distributed "
                f"across output channels, thus memory calculation may be inaccurate due to "
                f"SIMD assumptions.")
            nparams_per_oc = np.ceil(nparams_per_oc)
            # assert int(nparams_per_oc)==nparams_per_oc, f"Expected number of params per channel to be integer but is {nparams_per_oc}"

        num_oc_with_null_channels = np.ceil(num_oc / node_simd) * node_simd
        return num_oc_with_null_channels * nparams_per_oc
