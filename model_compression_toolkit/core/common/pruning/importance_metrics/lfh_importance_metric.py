from typing import Callable, List

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.hessian import HessianInfoService, HessianMode, HessianInfoGranularity
from model_compression_toolkit.core.common.pruning.importance_metrics.base_importance_metric import BaseImportanceMetric
import numpy as np

from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig


class LFHImportanceMetric(BaseImportanceMetric):

    def __init__(self,
                 graph:Graph,
                 representative_data_gen: Callable,
                 fw_impl: FrameworkImplementation,
                 pruning_config: PruningConfig,
                 fw_info: FrameworkInfo):

        self.float_graph= graph
        self.representative_data_gen = representative_data_gen
        self.fw_impl = fw_impl
        self.pruning_config = pruning_config
        self.fw_info = fw_info

    def get_entry_node_to_score(self, sections_input_nodes:List[BaseNode]):
        # Initialize services and variables for pruning process.
        hessian_info_service = HessianInfoService(graph=self.float_graph,
                                                  representative_dataset=self.representative_data_gen,
                                                  fw_impl=self.fw_impl)

        # Calculate the LFH (Label-Free Hessian) score for each prunable channel.
        scores_per_prunable_node = hessian_info_service.fetch_scores_for_multiple_nodes(
            mode=HessianMode.WEIGHTS,
            granularity=HessianInfoGranularity.PER_OUTPUT_CHANNEL,
            nodes=sections_input_nodes,
            required_size=self.pruning_config.num_score_approximations)


        # Average the scores across approximations and map them to the corresponding nodes.
        entry_node_to_score = {node: np.mean(scores, axis=0) for node, scores in
                               zip(sections_input_nodes, scores_per_prunable_node)}

        l2_oc_norm = self.get_l2_out_channel_norm(entry_nodes=sections_input_nodes)
        count_oc_nparams = self.count_oc_nparams(entry_nodes=sections_input_nodes)
        entry_node_to_score = self.normalize_lfh_scores(entry_node_to_score=entry_node_to_score,
                                                        entry_node_to_l2norm=l2_oc_norm,
                                                        entry_node_to_nparmas=count_oc_nparams)
        return entry_node_to_score


    def normalize_lfh_scores(self,
                             entry_node_to_score,
                             entry_node_to_l2norm,
                             entry_node_to_nparmas):
        new_scores = {}
        for node, trace_vector in entry_node_to_score.items():
            new_scores[node] = trace_vector*entry_node_to_l2norm[node]/entry_node_to_nparmas[node]
        return new_scores

    def count_oc_nparams(self, entry_nodes: List[BaseNode]):
        node_channel_params = {}
        for entry_node in entry_nodes:
            kernel = entry_node.get_weights_by_keys('kernel')
            ox_axis = self.fw_info.kernel_channels_mapping.get(entry_node.type)[0]

            # Calculate the number of parameters for each output channel
            params_per_channel = np.prod(kernel.shape) / kernel.shape[ox_axis]

            # Create an array with the number of parameters per channel
            num_params_array = np.full(kernel.shape[ox_axis], params_per_channel)

            # Store in node_channel_params a dictionary from node to a np.array where
            # each element corresponds to the number of parameters of this channel
            node_channel_params[entry_node] = num_params_array

        return node_channel_params


    def get_l2_out_channel_norm(self, entry_nodes: List[BaseNode]):
        node_l2_channel_norm = {}
        for entry_node in entry_nodes:
            kernel = entry_node.get_weights_by_keys('kernel')
            ox_axis = self.fw_info.kernel_channels_mapping.get(entry_node.type)[0]

            # Compute the l2 norm of each output channel
            channels = np.split(kernel, indices_or_sections=kernel.shape[ox_axis], axis=ox_axis)
            l2_norms = [np.linalg.norm(c.flatten(), ord=2) ** 2 for c in channels]

            # Store in node_l2_channel_norm a dictionary from node to a np.array where
            # each element corresponds to the l2 norm of this channel
            node_l2_channel_norm[entry_node] = l2_norms

        return node_l2_channel_norm