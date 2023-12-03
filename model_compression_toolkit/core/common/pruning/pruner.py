import numpy as np
from typing import Callable, List, Dict

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.hessian import HessianInfoService, HessianMode, HessianInfoGranularity
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.pruning.greedy_mask_calculator import GreedyMaskCalculator
from model_compression_toolkit.core.common.pruning.importance_metrics.importance_metric_factory import \
    get_importance_metric
from model_compression_toolkit.core.common.pruning.prune_graph import build_pruned_graph
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig, ChannelsFilteringStrategy, \
    ImportanceMetric
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

        self.mask = None # Will hold the output-channel mask for each entry node.
        self.scores = None  # Will hold the importance scores for each entry node.

    def get_pruned_graph(self) -> Graph:
        """
        Prunes the graph according to the specified KPI and pruning configuration.

        Returns:
            Graph: The pruned computational graph.
        """
        # Retrieve entry points from the graph.
        entry_nodes = self.float_graph.get_pruning_sections_entry_nodes(self.fw_info, self.fw_impl)

        # Compute the importance score for each entry node in the graph.
        self.scores = self.get_score_per_entry_point(entry_nodes)

        # Assert that scores were successfully computed.
        assert self.scores is not None

        # Choose the mask calculation strategy based on the pruning configuration.
        if self.pruning_config.channels_filtering_strategy == ChannelsFilteringStrategy.GREEDY:
            # Instantiate a GreedyMaskCalculator to compute the pruning masks.
            mask_calculator = GreedyMaskCalculator(entry_nodes,
                                                   self.fw_info,
                                                   self.scores,
                                                   self.target_kpi,
                                                   self.float_graph,
                                                   self.fw_impl,
                                                   self.target_platform_capabilities)

            # Calculate the mask that will be used to prune the graph.
            self.mask = mask_calculator.get_mask()
        else:
            # Log an error if an unsupported channel filtering strategy is specified.
            Logger.error("Currently, only the GREEDY ChannelsFilteringStrategy is supported.")

        # Construct the pruned graph using the computed masks.
        pruned_graph = build_pruned_graph(self.float_graph,
                                          self.mask,
                                          self.fw_info,
                                          self.fw_impl)

        # Return the pruned computational graph.
        return pruned_graph

    def get_score_per_entry_point(self, sections_input_nodes):
        """
        Computes the importance scores for each entry node that is eligible for pruning.

        Args:
            sections_input_nodes (List[BaseNode]): List of nodes that are the input points for pruning sections.

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

        entry_node_to_score = im.get_entry_node_to_score(sections_input_nodes)

        # Return the dictionary of nodes mapped to their importance scores.
        return entry_node_to_score


    def get_pruning_info(self) -> PruningInfo:
        """
        Gathers and returns pruning information including masks and scores.

        Returns:
            PruningInfo: An object containing the masks used for pruning and the importance scores of channels.
        """
        # Create and return a PruningInfo object with the collected pruning data.
        info = PruningInfo(pruning_masks=self.mask,
                           importance_scores=self.scores)
        return info
