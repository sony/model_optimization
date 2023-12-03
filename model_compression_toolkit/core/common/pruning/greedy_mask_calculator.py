import numpy as np
from typing import List, Dict, Tuple

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.pruning.memory_calculator import MemoryCalculator
from model_compression_toolkit.logger import Logger

class GreedyMaskCalculator:
    """
    GreedyMaskCalculator computes pruning masks for each prunable node in the graph
    using a greedy algorithm, aiming to meet a target KPI for memory footprint.
    """
    def __init__(self,
                 prunable_nodes: List[BaseNode],
                 fw_info: FrameworkInfo,
                 score_by_node: Dict[BaseNode, np.ndarray],
                 target_kpi: KPI,
                 graph,
                 fw_impl,
                 tpc):
        """
        Initializes the GreedyMaskCalculator with the required information.

        Args:
            prunable_nodes: List of nodes that can be pruned.
            fw_info: Framework-specific information and utilities.
            score_by_node: A dictionary mapping nodes to their importance scores.
            target_kpi: Target KPI to achieve after pruning.
            graph: The computational graph of the model.
            fw_impl: Framework-specific implementation details.
        """
        self.prunable_nodes = prunable_nodes
        self.fw_info = fw_info
        self.score_by_node = score_by_node
        self.target_kpi = target_kpi
        self.graph = graph
        self.fw_impl = fw_impl
        self.tpc = tpc

        # Initialize the SIMD group indices and scores dictionaries.
        self.simd_groups_indices = {} # TODO: Take SIMD grouping out of mask calculator
        self.simd_groups_scores = {}
        self.mask_simd = None  # Will hold SIMD group mask per node.
        self.mask = None  # Will hold the final mask to be applied to the nodes.

        # MemoryCalculator object to estimate the memory footprint of the pruned graph.
        self.memory_calculator = MemoryCalculator(graph=graph,
                                                  fw_info=fw_info,
                                                  fw_impl=fw_impl)

    def get_mask(self) -> Dict[BaseNode, np.ndarray]:
        """
        Gets the pruning mask for the graph. If the mask has not been computed yet,
        it triggers the computation.

        Returns:
            A dictionary mapping each prunable node to its computed pruning mask.
        """
        if self.mask is None:
            self._compute_mask()
        return self.mask

    def _compute_mask(self):
        """
        Computes the mask for each prunable node in the graph, based on their
        importance scores and the target KPI.
        """
        # Initialize masks for each node based on the number of output channels and SIMD groups.
        self.init_masks()

        # Group scores by SIMD size and set the first group to be unpruned.
        self.group_scores_by_simd_groups()
        # TODO: recalculate new score to each group instead of summing it _get_best_simd_group_candidate

        # Set the first group of channels to be unpruned.
        self.update_mandatory_mask()

        # Iteratively prune the graph while monitoring the memory footprint.
        current_memory = self.memory_calculator.get_pruned_graph_memory(masks=self.mask,
                                                                            include_padded_channels=self.tpc.is_simd_padding())
        if current_memory > self.target_kpi.weights_memory:
            Logger.error(f"Minimal required memory is {current_memory}, but target KPI is {self.target_kpi.weights_memory}")

        # Greedily add groups to the mask until the memory target is met or all channels remains.
        while current_memory < self.target_kpi.weights_memory and self.has_pruned_channel():
            # Select the best SIMD group to add based on the scores.
            node_to_remain, group_to_remain_idx = self._get_best_simd_group_candidate()
            self._update_simd_mask(node=node_to_remain, group_index=group_to_remain_idx, value=1)
            current_memory = self.memory_calculator.get_pruned_graph_memory(masks=self.mask,
                                                                                include_padded_channels=self.tpc.is_simd_padding())

        # If the target memory is exceeded, revert the last addition.
        if current_memory > self.target_kpi.weights_memory:
            self._update_simd_mask(node=node_to_remain, group_index=group_to_remain_idx, value=0)

    def group_scores_by_simd_groups(self):
        for prunable_node, node_scores in self.score_by_node.items():
            # Group scores and indices by SIMD size.
            self.simd_groups_scores[prunable_node], self.simd_groups_indices[prunable_node] = self._group_scores_by_simd_size(node_scores, prunable_node.get_simd())

    def update_mandatory_mask(self):
        for prunable_node, node_scores in self.score_by_node.items():
            # Set the first group of channels to be unpruned by default.
            self._update_simd_mask(node=prunable_node, group_index=0, value=1)

    def _update_simd_mask(self,
                          node: BaseNode,
                          group_index: int,
                          value: int):
        """
        Updates the mask for a specific SIMD group of a node.

        Args:
            node: The prunable node for which the mask is to be updated.
            group_index: Index of the SIMD group within the node.
            value: The new value (0 or 1) to set for the group mask.
        """
        assert value in [0, 1], "Mask value must be either 0 or 1."
        self.mask_simd[node][group_index] = value
        node_mask_indices = self.simd_groups_indices[node][group_index]
        self.mask[node][node_mask_indices] = value

    def init_masks(self):
        self.mask = {}
        self.mask_simd = {}
        for prunable_node in self.prunable_nodes:
            # Get the number of output channels and initialize the masks.
            num_oc = prunable_node.get_weights_by_keys(self.fw_info.get_kernel_op_attributes(prunable_node.type)[0]).shape[self.fw_info.kernel_channels_mapping.get(prunable_node.type)[0]]
            layer_mask = np.zeros(num_oc)
            layer_num_simd_groups = int(max(np.ceil(num_oc / prunable_node.get_simd()), 1))
            layer_mask_per_simd_group = np.zeros(layer_num_simd_groups)

            # Store the initialized masks.
            self.mask_simd[prunable_node] = layer_mask_per_simd_group
            self.mask[prunable_node] = layer_mask

    def _group_scores_by_simd_size(self, scores: np.ndarray, simd: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Groups the scores and their corresponding indices based on SIMD size.

        Args:
            scores: An array of scores to be grouped.
            simd: Size of the SIMD group.

        Returns:
            A tuple of two lists:
            - The first list contains arrays of scores grouped by SIMD size.
            - The second list contains arrays of corresponding indices for the score groups.
        """
        # Sort scores in descending order and get the corresponding indices.
        sorted_indices = np.argsort(-scores)

        # Calculate the number of complete groups.
        num_complete_groups = len(scores) // simd

        # Group scores and indices based on SIMD size.
        scores_groups = [scores[sorted_indices[i * simd:(i + 1) * simd]] for i in range(num_complete_groups)]
        indices_groups = [sorted_indices[i * simd:(i + 1) * simd] for i in range(num_complete_groups)]

        # Handle the remaining scores if they don't perfectly divide by simd size.
        remainder = len(scores) % simd
        if remainder:
            scores_groups.append(scores[sorted_indices[-remainder:]])
            indices_groups.append(sorted_indices[-remainder:])

        return scores_groups, indices_groups

    def _get_best_simd_group_candidate(self) -> Tuple[BaseNode, int]:
        """
        Finds the best SIMD group candidate for pruning.

        Returns:
            A tuple containing the node with the best SIMD group and the group index.
        """
        # Initialize variables to track the best score and corresponding node and group index.
        best_score = -np.inf
        best_node = None
        best_group_idx = -1

        for node, mask in self.mask_simd.items():
            # Get the index of the first zero in the mask. A zero indicates a prunable channel group.
            group_idx = int(np.argmax(mask == 0))

            # If group_idx is 0, it means there are no zeros in the mask, so this group is not prunable.
            if group_idx != 0:
                score = np.sum(self.simd_groups_scores[node][group_idx])
                # If the score for this group is better than the best score found so far, update the best score.
                if score > best_score:
                    best_score = score
                    best_node = node
                    best_group_idx = group_idx

        # If no prunable group was found, best_node will remain None.
        if best_node is None:
            raise ValueError("No prunable SIMD group found.")

        return best_node, best_group_idx


    def has_pruned_channel(self):
        """
        Checks if there is at least one channel marked for pruning in any node mask.

        Returns:
            True if there is at least one channel to be pruned, False otherwise.
        """
        return any(0 in mask for mask in self.mask.values())
