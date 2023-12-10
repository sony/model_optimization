from typing import Callable, List

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.pruning.channels_grouping import ChannelGrouping
from model_compression_toolkit.core.common.pruning.importance_metrics.base_importance_metric import BaseImportanceMetric
import numpy as np

from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig


class RandomImportanceMetric(BaseImportanceMetric):

    def __init__(self,
                 graph: Graph,
                 representative_data_gen: Callable,
                 fw_impl: FrameworkImplementation,
                 pruning_config: PruningConfig,
                 fw_info: FrameworkInfo):
        self.float_graph = graph
        self.representative_data_gen = representative_data_gen
        self.fw_impl = fw_impl
        self.pruning_config = pruning_config
        self.fw_info = fw_info


    def get_entry_node_to_simd_score(self, entry_nodes: List[BaseNode]):
        entry_node_to_score = self._get_entry_node_to_score(entry_nodes)
        self.channel_grouping = ChannelGrouping(prunable_nodes=entry_nodes,
                                                fw_info=self.fw_info)
        self.channel_grouping.group_scores_by_simd_groups(entry_node_to_score)
        grouped_indices = self.channel_grouping.get_group_indices()
        entry_node_to_simd_score = {}
        for node, trace in entry_node_to_score.items():
            trace_by_group = [np.sum(trace[g]) for g in grouped_indices[node]]
            entry_node_to_simd_score[node]=np.asarray(trace_by_group)
        return entry_node_to_simd_score, grouped_indices


    def _get_entry_node_to_score(self, sections_input_nodes: List[BaseNode]):
        random_scores = [np.random.random(
            node.get_weights_by_keys('kernel').shape[self.fw_info.kernel_channels_mapping.get(node.type)[0]])
            for node in sections_input_nodes]
        entry_node_to_score = {node: scores for node, scores in zip(sections_input_nodes, random_scores)}
        return entry_node_to_score