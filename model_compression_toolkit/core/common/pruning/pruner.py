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
from typing import Callable, List, Dict

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.pruning.greedy_mask_calculator import GreedyMaskCalculator
from model_compression_toolkit.core.common.pruning.importance_metrics.importance_metric_factory import \
    get_importance_metric
from model_compression_toolkit.core.common.pruning.prune_graph import build_pruned_graph
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig, ChannelsFilteringStrategy
from model_compression_toolkit.core.common.pruning.pruning_info import PruningInfo
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
import time

class Pruner:
    """
    Pruner class responsible for applying pruning to a computational graph to meet a target KPI.
    """
    def __init__(self,
                 float_graph: Graph,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation,
                 target_kpi: KPI,
                 representative_data_gen: Callable,
                 pruning_config: PruningConfig,
                 target_platform_capabilities: TargetPlatformCapabilities):
        """
        Initializes the Pruner instance with the necessary information and configurations.

        Args:
            float_graph (Graph): The floating-point representation of the model's computation graph.
            fw_info (FrameworkInfo): Contains metadata and helper functions for the framework.
            fw_impl (FrameworkImplementation): Implementation of specific framework methods required for pruning.
            target_kpi (KPI): The target KPIs to be achieved after pruning.
            representative_data_gen (Callable): Generator function for representative dataset used in pruning analysis.
            pruning_config (PruningConfig): Configuration object specifying how pruning should be performed.
            target_platform_capabilities (TargetPlatformCapabilities): Object encapsulating the capabilities of the target hardware platform.
        """
        # Initialize member variables with provided arguments.
        self.float_graph = float_graph
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.target_kpi = target_kpi
        self.representative_data_gen = representative_data_gen
        self.pruning_config = pruning_config
        self.target_platform_capabilities = target_platform_capabilities

        self._pruned_graph = None

        self.mask = None # Will hold the output-channel mask for each entry node.

        self.simd_scores = None  # Will hold the importance scores for each entry node.
        self.simd_groups_indices = None


    def get_pruned_graph(self) -> Graph:
        if not self._pruned_graph:
            self._create_pruned_graph()

        return self._pruned_graph

    def _create_pruned_graph(self):
        # Retrieve entry points from the graph.
        entry_nodes = self.float_graph.get_pruning_sections_entry_nodes(self.fw_info, self.fw_impl)

        # Compute the importance score for each entry node in the graph.
        self.simd_scores, self.simd_groups_indices = self.get_score_per_entry_point(entry_nodes)

        Logger.info(f"Computing pruning mask... This may take a while if the model is big and SIMD size is small")

        # Choose the mask calculation strategy based on the pruning configuration.
        if self.pruning_config.channels_filtering_strategy == ChannelsFilteringStrategy.GREEDY:
            # Instantiate a GreedyMaskCalculator to compute the pruning masks.
            mask_calculator = GreedyMaskCalculator(entry_nodes,
                                                   self.fw_info,
                                                   self.simd_scores,
                                                   self.target_kpi,
                                                   self.float_graph,
                                                   self.fw_impl,
                                                   self.target_platform_capabilities,
                                                   self.simd_groups_indices)

            # Calculate the mask that will be used to prune the graph.
            mask_calculator.compute_mask()
            self.mask = mask_calculator.get_mask()
        else:
            Logger.error("Currently, only the GREEDY ChannelsFilteringStrategy is supported.")

        Logger.info(f"Start pruning graph...")

        # Construct the pruned graph using the computed masks.
        self._pruned_graph = build_pruned_graph(self.float_graph,
                                                self.mask,
                                                self.fw_info,
                                                self.fw_impl)


    def get_score_per_entry_point(self, entry_nodes: List[BaseNode]):
        """
        Computes the importance scores for each entry node that is eligible for pruning.

        Args:
            entry_nodes (List[BaseNode]): List of nodes that are the input points for pruning sections.

        Returns:
            Dict[BaseNode, np.ndarray]: A dictionary mapping each entry node to its corresponding importance score.
        """
        # Initialize a dictionary to hold scores for each node.
        im = get_importance_metric(self.pruning_config.importance_metric,
                                   graph=self.float_graph,
                                   representative_data_gen=self.representative_data_gen,
                                   fw_impl=self.fw_impl,
                                   pruning_config=self.pruning_config,
                                   fw_info=self.fw_info)

        entry_node_to_simd_score, simd_groups_indices = im.get_entry_node_to_simd_score(entry_nodes)

        # Return the dictionary of nodes mapped to their importance scores.
        return entry_node_to_simd_score, simd_groups_indices


    def get_pruning_info(self) -> PruningInfo:
        """
        Gathers and returns pruning information including masks and scores.

        Returns:
            PruningInfo: An object containing the masks used for pruning and the importance scores of channels.
        """
        # Assert that scores were successfully computed.
        assert self.simd_scores is not None

        _scores = {}

        for node, groups_indices in self.simd_groups_indices.items():
            # Retrieve the original scores for the node
            node_scores = self.simd_scores[node]

            # Determine the total number of indices (and thus the length of the new score list)
            total_indices = sum(len(group) for group in groups_indices)

            # Create an array to store the new scores, initially filled with zeros
            new_node_scores = np.zeros(total_indices)

            # Assign the duplicated scores to the appropriate positions
            for group_score, group_indices in zip(node_scores, groups_indices):
                for index in group_indices:
                    new_node_scores[index] = group_score

            # Store the new scores in the dictionary
            _scores[node] = new_node_scores

        # Create and return a PruningInfo object with the collected pruning data.
        info = PruningInfo(pruning_masks=self.mask,
                           importance_scores=_scores)
        return info
