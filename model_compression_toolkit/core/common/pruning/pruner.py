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
from typing import Callable, List, Dict, Tuple

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core.common.pruning.greedy_mask_calculator import GreedyMaskCalculator
from model_compression_toolkit.core.common.pruning.importance_metrics.importance_metric_factory import \
    get_importance_metric
from model_compression_toolkit.core.common.pruning.prune_graph import build_pruned_graph
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig, ChannelsFilteringStrategy
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import \
    PruningFrameworkImplementation
from model_compression_toolkit.core.common.pruning.pruning_info import PruningInfo, \
    unroll_simd_scores_to_per_channel_scores
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework import \
    FrameworkQuantizationCapabilities


class Pruner:
    """
    Pruner class responsible for applying pruning to a computational graph to meet a target resource utilization.
    It identifies and prunes less significant channels based on importance scores, considering SIMD constraints.
    """
    def __init__(self,
                 float_graph: Graph,
                 fw_impl: PruningFrameworkImplementation,
                 target_resource_utilization: ResourceUtilization,
                 representative_data_gen: Callable,
                 pruning_config: PruningConfig,
                 target_platform_capabilities: FrameworkQuantizationCapabilities):
        """
        Args:
            float_graph (Graph): The floating-point representation of the model's computation graph.
            fw_impl (PruningFrameworkImplementation): Implementation of specific framework methods required for pruning.
            target_resource_utilization (ResourceUtilization): The target resource utilization to be achieved after pruning.
            representative_data_gen (Callable): Generator function for representative dataset used in pruning analysis.
            pruning_config (PruningConfig): Configuration object specifying how pruning should be performed.
            target_platform_capabilities (FrameworkQuantizationCapabilities): Object encapsulating the capabilities of the target hardware platform.
        """
        self.float_graph = float_graph
        self.fw_impl = fw_impl
        self.target_resource_utilization = target_resource_utilization
        self.representative_data_gen = representative_data_gen
        self.pruning_config = pruning_config
        self.target_platform_capabilities = target_platform_capabilities

        # Internal variables for storing the pruned graph and intermediate data.
        self.per_oc_mask = None  # Output-channel mask for each entry node.
        self.simd_scores = None  # Importance scores considering SIMD groups.
        self.simd_groups_indices = None  # Indices of SIMD groups in each node.

    def prune_graph(self):
        """
        Main method for pruning the graph. Computes importance scores, calculates pruning masks,
        and constructs the pruned graph based on these masks.
        """
        # Fetch entry nodes and compute importance scores for SIMD groups.
        entry_nodes = self.float_graph.get_pruning_sections_entry_nodes(self.fw_impl)
        self.simd_scores, self.simd_groups_indices = self.get_score_per_entry_point(entry_nodes)

        Logger.info(f"Calculating the pruning mask. Please note that this process might take some time,"
                    f" especially for large models or when using a small SIMD size.")

        # Apply Greedy strategy to compute masks based on importance scores.
        if self.pruning_config.channels_filtering_strategy == ChannelsFilteringStrategy.GREEDY:
            mask_calculator = GreedyMaskCalculator(entry_nodes,
                                                   self.simd_scores,
                                                   self.target_resource_utilization,
                                                   self.float_graph,
                                                   self.fw_impl,
                                                   self.target_platform_capabilities,
                                                   self.simd_groups_indices)
            mask_calculator.compute_mask()
            self.per_oc_mask = mask_calculator.get_mask()
        else:
            Logger.critical("Only GREEDY ChannelsFilteringStrategy is currently supported.")  # pragma: no cover

        Logger.info("Start pruning graph...")
        _pruned_graph = build_pruned_graph(self.float_graph,
                                           self.per_oc_mask,
                                           self.fw_impl)
        return _pruned_graph

    def get_score_per_entry_point(self, entry_nodes: List[BaseNode]) -> Tuple[Dict[BaseNode, np.ndarray], Dict[BaseNode, List[np.ndarray]]]:
        """
        Calculates the importance score for each entry node in the graph.

        Args:
            entry_nodes (List[BaseNode]): List of entry nodes in the graph.

        Returns:
            Tuple: Tuple containing importance scores and group indices.
        """
        # Retrieve and initialize the importance metric.
        im = get_importance_metric(self.pruning_config.importance_metric, graph=self.float_graph,
                                   representative_data_gen=self.representative_data_gen, fw_impl=self.fw_impl,
                                   pruning_config=self.pruning_config)
        entry_node_to_simd_score, simd_groups_indices = im.get_entry_node_to_simd_score(entry_nodes)
        return entry_node_to_simd_score, simd_groups_indices

    def get_pruning_info(self) -> PruningInfo:
        """
        Compiles and returns detailed pruning information, including masks and channel scores.

        Returns:
            PruningInfo: Object containing detailed pruning data.
        """
        # Convert SIMD group scores to per-channel scores and create PruningInfo.
        _per_oc_scores = unroll_simd_scores_to_per_channel_scores(self.simd_scores, self.simd_groups_indices)
        info = PruningInfo(pruning_masks=self.per_oc_mask, importance_scores=_per_oc_scores)
        return info

