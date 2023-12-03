from typing import Callable, List

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.hessian import HessianInfoService, HessianMode, HessianInfoGranularity
from model_compression_toolkit.core.common.pruning.importance_metrics.base_importance_metric import BaseImportanceMetric
import numpy as np

from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import PruningFrameworkImplementation


class LFHImportanceMetric(BaseImportanceMetric):
    def __init__(self,
                 graph: Graph,
                 representative_data_gen: Callable,
                 fw_impl: PruningFrameworkImplementation,
                 pruning_config: PruningConfig,
                 fw_info: FrameworkInfo):
        """
        Initializes the LFHImportanceMetric class for calculating Label-Free-Hessian
        based importance scores of nodes in a graph.

        Args:
            graph (Graph): The computational graph of the neural network.
            representative_data_gen (Callable): A generator function to produce representative data for the network.
            fw_impl (PruningFrameworkImplementation): The specific framework implementation.
            pruning_config (PruningConfig): Configuration settings for the pruning process.
            fw_info (FrameworkInfo): Framework-specific information and utilities.
        """
        self.float_graph = graph
        self.representative_data_gen = representative_data_gen
        self.fw_impl = fw_impl
        self.pruning_config = pruning_config
        self.fw_info = fw_info

    def get_entry_node_to_score(self, sections_input_nodes: List[BaseNode]):
        """
        Calculates importance scores for each entry node in the provided list using the LFH method.

        Args:
            sections_input_nodes (List[BaseNode]): List of entry nodes for which to calculate importance scores.

        Returns:
            Dict[BaseNode, np.ndarray]: A dictionary mapping each entry node to its importance scores.
        """
        # Initialize Hessian information service to calculate LFH scores
        hessian_info_service = HessianInfoService(graph=self.float_graph,
                                                  representative_dataset=self.representative_data_gen,
                                                  fw_impl=self.fw_impl)

        # Fetch scores for multiple nodes in the graph
        scores_per_prunable_node = hessian_info_service.fetch_scores_for_multiple_nodes(mode=HessianMode.WEIGHTS,
                                                                                        granularity=HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                                                                                        nodes=sections_input_nodes,
                                                                                        required_size=self.pruning_config.num_score_approximations)

        # Average scores across approximations and map them to nodes
        entry_node_to_score = {node: np.mean(scores, axis=0) for node, scores in
                               zip(sections_input_nodes, scores_per_prunable_node)}

        # Normalize scores using L2 norms and number of parameters
        l2_oc_norm = self.get_l2_out_channel_norm(entry_nodes=sections_input_nodes)
        count_oc_nparams = self.count_oc_nparams(entry_nodes=sections_input_nodes)
        entry_node_to_score = self.normalize_lfh_scores(entry_node_to_score, l2_oc_norm, count_oc_nparams)
        return entry_node_to_score

    def normalize_lfh_scores(self, entry_node_to_score, entry_node_to_l2norm, entry_node_to_nparmas):
        """
        Normalizes the LFH scores for each node.

        Args:
            entry_node_to_score (Dict[BaseNode, np.ndarray]): Dictionary of nodes and their LFH scores.
            entry_node_to_l2norm (Dict[BaseNode, np.ndarray]): Dictionary of nodes and their L2 norms.
            entry_node_to_nparmas (Dict[BaseNode, np.ndarray]): Dictionary of nodes and their parameter counts.

        Returns:
            Dict[BaseNode, np.ndarray]: Normalized scores for each node.
        """
        new_scores = {}
        # Normalize scores by multiplying with L2 norm and dividing by number of parameters
        for node, trace_vector in entry_node_to_score.items():
            new_scores[node] = trace_vector * entry_node_to_l2norm[node] / entry_node_to_nparmas[node]
        return new_scores

    def count_oc_nparams(self, entry_nodes: List[BaseNode]):
        """
        Counts the number of parameters per output channel for each entry node.

        Args:
            entry_nodes (List[BaseNode]): List of entry nodes to count parameters for.

        Returns:
            Dict[BaseNode, np.ndarray]: Dictionary of nodes and their parameters count per output channel.
        """
        node_channel_params = {}
        for entry_node in entry_nodes:
            kernel = entry_node.get_weights_by_keys('kernel')
            oc_axis = self.fw_info.kernel_channels_mapping.get(entry_node.type)[0]

            # Calculate parameters per channel
            params_per_channel = np.prod(kernel.shape) / kernel.shape[oc_axis]
            num_params_array = np.full(kernel.shape[oc_axis], params_per_channel)

            node_channel_params[entry_node] = num_params_array
        return node_channel_params

    def get_l2_out_channel_norm(self, entry_nodes: List[BaseNode]):
        """
        Computes the L2 norm for each output channel of the entry nodes.

        Args:
            entry_nodes (List[BaseNode]): List of entry nodes for L2 norm computation.

        Returns:
            Dict[BaseNode, np.ndarray]: Dictionary of nodes and their L2 norms for each output channel.
        """
        node_l2_channel_norm = {}
        for entry_node in entry_nodes:
            kernel = entry_node.get_weights_by_keys('kernel')
            ox_axis = self.fw_info.kernel_channels_mapping.get(entry_node.type)[0]

            # Compute L2 norm for each channel
            channels = np.split(kernel, indices_or_sections=kernel.shape[ox_axis], axis=ox_axis)
            l2_norms = [np.linalg.norm(c.flatten(), ord=2) ** 2 for c in channels]

            node_l2_channel_norm[entry_node] = l2_norms
        return node_l2_channel_norm
