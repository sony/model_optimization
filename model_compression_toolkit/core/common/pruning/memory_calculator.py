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
from typing import List, Dict

from model_compression_toolkit.constants import FP32_BYTES_PER_PARAMETER
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import \
    PruningFrameworkImplementation
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSection, PruningSectionMask
from model_compression_toolkit.logger import Logger


class MemoryCalculator:
    """
    MemoryCalculator is used for estimating the memory usage of a graph under pruning mask.
    It takes into account the specific pruning masks applied to each node in the network,
    including handling of shared nodes between pruning sections and consideration of SIMD-padded channels.
    The calculator aids in understanding the impact of pruning on the overall memory footprint of the model,
    which is crucial for deploying models on memory-constrained devices or optimizing for computational efficiency.
    """

    def __init__(self, graph: Graph, fw_info: FrameworkInfo, fw_impl: PruningFrameworkImplementation):
        """
        Initializes the MemoryCalculator with necessary information about the model's graph,
        framework-specific details, and pruning implementation.

        Args:
            graph (Graph): Computational graph of the model.
            fw_info (FrameworkInfo): Contains framework-specific information.
            fw_impl (PruningFrameworkImplementation): Implementation details for pruning.
        """
        self.graph = graph
        self.fw_info = fw_info
        self.fw_impl = fw_impl

    def get_pruned_graph_memory(self,
                                masks: Dict[BaseNode, np.ndarray],
                                include_padded_channels: bool) -> float:
        """
        Calculates the memory usage of the pruned graph.

        Args:
            masks (Dict[BaseNode, np.ndarray]): Dictionary mapping nodes to their pruning masks.
            include_padded_channels (bool): Whether to include padded channels in the memory calculation.

        Returns:
            float: Estimated memory usage of the pruned graph in bytes.
        """
        nparams = self.get_pruned_graph_num_params(masks, include_padded_channels)
        return nparams * FP32_BYTES_PER_PARAMETER  # Assuming each parameter is 4 bytes (float32)

    def get_pruned_graph_num_params(self,
                                    masks: Dict[BaseNode, np.ndarray],
                                    include_padded_channels: bool) -> int:
        """
        Calculates the total number of parameters in the pruned graph.

        Args:
            masks (Dict[BaseNode, np.ndarray]): Pruning masks for each node.
            include_padded_channels (bool): Flag to include SIMD-padded channels in the count.

        Returns:
            int: Total number of parameters in the pruned graph.
        """
        total_nparams = 0

        pruning_sections = self.graph.get_pruning_sections(self.fw_impl)
        total_nparams += self.get_nparams_of_nonpruned_nodes(pruning_sections, include_padded_channels)
        total_nparams += self.get_nparams_of_pruning_sections(masks, pruning_sections, include_padded_channels)
        total_nparams -= self.get_nparams_of_shared_nodes(masks, pruning_sections, include_padded_channels)

        return total_nparams

    def get_nparams_of_shared_nodes(self,
                                    masks: Dict[BaseNode, np.ndarray],
                                    pruning_sections: List[PruningSection],
                                    include_padded_channels: bool) -> int:
        """
        Calculate the number of parameters for nodes shared between adjacent pruning sections.

        Args:
            masks (Dict[BaseNode, np.ndarray]): Pruning masks for each node.
            pruning_sections (List[PruningSection]): A list of pruning sections.
            include_padded_channels (bool): Flag to include padded channels in the count.

        Returns:
            int: Total number of parameters for shared nodes.
        """
        nparams = 0
        shared_nodes = self._get_nodes_from_adjacent_sections(pruning_sections)
        for node in shared_nodes:
            node_input_mask = self._get_exit_node_input_mask(node, pruning_sections, masks)
            node_output_mask = masks.get(node)
            nparams += self.get_pruned_node_num_params(node, node_input_mask, node_output_mask, include_padded_channels)
        return nparams

    def get_nparams_of_pruning_sections(self,
                                        masks: Dict[BaseNode, np.ndarray],
                                        pruning_sections: List[PruningSection],
                                        include_padded_channels: bool) -> int:
        """
        Calculate the number of parameters for all pruning sections.

        Args:
            masks (dict): Pruning masks for each node.
            pruning_sections (list): A list of pruning sections.
            include_padded_channels (bool): Flag to include padded channels in the count.

        Returns:
            int: Total number of parameters for all pruning sections.
        """
        nparams = 0
        for pruning_section in pruning_sections:
            pruning_section_mask = self.get_section_mask_from_node_mask(masks, pruning_section, pruning_sections)
            nparams += self._get_pruning_section_num_params(pruning_section, pruning_section_mask,
                                                            include_padded_channels)
        return nparams

    def get_section_mask_from_node_mask(self,
                                        masks: Dict[BaseNode, np.ndarray],
                                        pruning_section: PruningSection,
                                        pruning_sections: List[PruningSection]) -> PruningSectionMask:
        """
        Create a pruning section mask from individual node masks.

        Args:
            masks (dict): Pruning masks for each node.
            pruning_section (PruningSection): The current pruning section.
            pruning_sections (list): A list of pruning sections.

        Returns:
            PruningSectionMask: The combined pruning mask for the section.
        """
        first_node_input_channels_mask = self._get_exit_node_input_mask(pruning_section.entry_node,
                                                                        pruning_sections,
                                                                        masks)
        second_node_output_mask = masks.get(pruning_section.exit_node)

        return PruningSectionMask(
            entry_node_ic_mask=first_node_input_channels_mask,
            entry_node_oc_mask=masks.get(pruning_section.entry_node),
            exit_node_ic_mask=masks.get(pruning_section.entry_node),
            exit_node_oc_mask=second_node_output_mask
        )

    def get_nparams_of_nonpruned_nodes(self,
                                       pruning_sections: List[PruningSection],
                                       include_padded_channels: bool) -> int:
        """
        Calculate the number of parameters for non-pruned nodes.

        Args:
            pruning_sections (list): A list of pruning sections.
            include_padded_channels (bool): Flag to include padded channels in the count.

        Returns:
            int: Total number of parameters for non-pruned nodes.
        """
        total_nparams = 0
        nodes_to_prune = set(node for section in pruning_sections for node in section.get_all_section_nodes())
        for n in self.graph.nodes:
            if n not in nodes_to_prune:
                node_nparams = self.get_pruned_node_num_params(n,
                                                               None,
                                                               None,
                                                               include_padded_channels)
                total_nparams += node_nparams
        return total_nparams

    def _get_exit_node_input_mask(self,
                                  node: BaseNode,
                                  pruning_sections: List[PruningSection],
                                  masks: Dict[BaseNode, np.ndarray]) -> np.ndarray:
        """
        Retrieves the input mask for an exit node based on the pruning sections.
        The function searches for the input channels mask of an exit node based on the output-channels mask
        of the corresponding entry node in the graph. If such mask is not found, a mask of 1s is returned.

        Args:
            node (BaseNode): The exit node for which the input mask is required.
            pruning_sections (List[PruningSection]): A list of pruning sections in the graph.
            masks (Dict[BaseNode, np.ndarray]): A dictionary mapping nodes to their respective pruning masks.

        Returns:
            np.ndarray: The input mask for the specified exit node, or 1s mask if not found.
        """
        for section in pruning_sections:
            # If the node is the exit node of a pruning section, return the entry node's mask.
            if node == section.exit_node:
                return masks.get(section.entry_node)

        kernel_attr = self.fw_info.get_kernel_op_attributes(node.type)
        # Ensure only one kernel attribute exists for the given node.
        if len(kernel_attr) != 1:
            Logger.critical(f"Expected a single attribute, but found {len(kernel_attr)} attributes for node '{node}'. Ensure the node configuration is correct.")
        kernel_attr = kernel_attr[0]

        # Retrieve and validate the axis index for the output channels.
        _, ic_axis = self.fw_info.kernel_channels_mapping.get(node.type)
        if ic_axis is None or int(ic_axis) != ic_axis:
            Logger.critical(f"Invalid input channel axis type for node '{node}': expected integer but got '{ic_axis}'.")

        # Get the number of output channels based on the kernel attribute and axis.
        num_ic = node.get_weights_by_keys(kernel_attr).shape[ic_axis]
        mask = np.ones(num_ic, dtype=bool)
        return mask

    def _get_nodes_from_adjacent_sections(self,
                                          pruning_sections: List[PruningSection]) -> List[BaseNode]:
        """
        Identifies nodes that are shared between adjacent pruning sections.

        Args:
            pruning_sections (List[PruningSection]): A list of pruning sections in the graph.

        Returns:
            List[BaseNode]: A list of nodes that are present at the boundaries of adjacent sections.
        """
        input_nodes = set(section.entry_node for section in pruning_sections)
        output_nodes = set(section.exit_node for section in pruning_sections)
        # Return the intersection of entry and exit nodes, which represents shared nodes.
        return list(input_nodes.intersection(output_nodes))

    def _get_pruning_section_num_params(self,
                                        pruning_section: PruningSection,
                                        pruning_section_mask: PruningSectionMask,
                                        include_padded_channels: bool) -> int:
        """
        Calculates the total number of parameters in a pruning section after applying the pruning mask.

        Args:
            pruning_section (PruningSection): The pruning section to be considered.
            pruning_section_mask (PruningSectionMask): The pruning mask applied to the section.
            include_padded_channels (bool): Flag to include padded channels in the count.

        Returns:
            int: The total number of parameters in the pruning section after pruning.
        """
        # Calculate the number of parameters for the entry node.
        first_node_nparams = self.get_pruned_node_num_params(pruning_section.entry_node,
                                                             pruning_section_mask.entry_node_ic_mask,
                                                             pruning_section_mask.entry_node_oc_mask,
                                                             include_padded_channels)

        # Sum the number of parameters for all intermediate nodes.
        total_inter_nodes_nparams = sum(
            self.get_pruned_node_num_params(inter_node, pruning_section_mask.entry_node_oc_mask,
                                            pruning_section_mask.entry_node_oc_mask, include_padded_channels) for
            inter_node in pruning_section.intermediate_nodes)

        # Calculate the number of parameters for the exit node.
        second_node_nparams = self.get_pruned_node_num_params(pruning_section.exit_node,
                                                              pruning_section_mask.exit_node_ic_mask,
                                                              pruning_section_mask.exit_node_oc_mask,
                                                              include_padded_channels)

        return first_node_nparams + total_inter_nodes_nparams + second_node_nparams

    def get_pruned_node_num_params(self,
                                   node: BaseNode,
                                   input_mask: np.ndarray,
                                   output_mask: np.ndarray,
                                   include_padded_channels: bool) -> int:
        """
        Calculates the number of parameters in a node after applying input and output pruning masks.

        Args:
            node (BaseNode): The node whose parameters are to be calculated.
            input_mask (np.ndarray): The mask applied to the input channels of the node.
            output_mask (np.ndarray): The mask applied to the output channels of the node.
            include_padded_channels (bool): Flag to include padded channels in the count due to SIMD.

        Returns:
            int: The total number of parameters in the node after pruning.
        """
        total_params = 0
        attributes_and_oc_axis = self.fw_impl.attrs_oi_channels_info_for_pruning(node, self.fw_info)

        # Iterate over the node's weights and apply pruning based on the masks.
        for w_attr, w in node.weights.items():
            io_axis = [io_axis for attr, io_axis in attributes_and_oc_axis.items() if attr in w_attr]
            if len(io_axis) != 1:
                Logger.critical(f"Each weight must correspond to exactly one IO (Input/Output) axis; however, the current configuration has '{io_axis}' axes.")
            out_axis, in_axis = io_axis[0]

            # Apply input and output masks to the weight tensor.
            if in_axis is not None and input_mask is not None:
                w = self._prune_tensor(w, input_mask, in_axis)
            if out_axis is not None and output_mask is not None:
                w = self._prune_tensor(w, output_mask, out_axis)

            total_params += w.size

        # Adjust the total parameter count if padded channels are to be included.
        if output_mask is not None:
            num_oc = np.sum(output_mask)
        else:
            # Get the node channel axis from framework info
            channel_axis = self.fw_info.out_channel_axis_mapping.get(node.type)
            if channel_axis is None:
                Logger.critical(f"The channel axis is undefined. Please ensure the channel axis is explicitly defined for node {node.type} in the framework info.")

            # Check if node.output_shape is a list of lists.
            # In this case make sure all the out channels are the same value
            if all(isinstance(sublist, list) for sublist in node.output_shape):
                compare_value = node.output_shape[0][channel_axis]
                if all(len(sublist) > channel_axis and sublist[channel_axis] == compare_value for sublist in node.output_shape):
                    num_oc = compare_value
                else:
                    Logger.critical("The number of output channels must be the same across all outputs of the node.")
            else:
                num_oc = node.output_shape[channel_axis]

        if include_padded_channels:
            total_params = self.get_node_nparams_with_padded_channels(node, total_params, num_oc, node.get_simd())

        return total_params

    def _prune_tensor(self,
                      w: np.ndarray,
                      mask: np.ndarray,
                      axis: int) -> np.ndarray:
        """
        Prunes a tensor along a specified axis using a provided mask.

        Args:
            w (np.ndarray): The weight tensor to be pruned.
            mask (np.ndarray): The pruning mask to apply.
            axis (int): The axis along which to apply the pruning mask.

        Returns:
            np.ndarray: The pruned tensor.
        """
        mask = np.ones(w.shape[axis], dtype=bool) if mask is None else mask.astype(bool)
        if w.shape[axis] != len(mask):
            Logger.critical(f"Expected a mask length of {len(mask)}, but got {w.shape[axis]}. Ensure the mask aligns with the tensor shape.")
        pruned_w = np.take(w, np.where(mask)[0], axis=axis)
        return pruned_w

    def get_node_nparams_with_padded_channels(self,
                                              node: BaseNode,
                                              node_nparams: int,
                                              num_oc: int,
                                              node_simd: int) -> int:
        """
        Adjusts the number of parameters of a node by considering padded channels due to SIMD.

        Args:
            node (BaseNode): The node whose parameters are being adjusted.
            node_nparams (int): The original number of parameters in the node.
            num_oc (int): The number of output channels in the node.
            node_simd (int): The SIMD width used in the node.

        Returns:
            The adjusted number of parameters considering padded channels.
        """
        if not (num_oc >= 1 and int(num_oc) == num_oc):
            Logger.critical(f"Expected the number of output channels to be a non-negative integer, but received '{num_oc}'.")

        nparams_per_oc = node_nparams / num_oc
        if int(nparams_per_oc) != nparams_per_oc:
            Logger.warning(
                f"Found a layer {node.name} with weights not uniformly distributed "
                f"across output channels; memory calculation may be inaccurate due to "
                f"SIMD assumptions.")
            nparams_per_oc = np.ceil(nparams_per_oc)

        num_oc_with_null_channels = np.ceil(num_oc / node_simd) * node_simd
        return num_oc_with_null_channels * nparams_per_oc
