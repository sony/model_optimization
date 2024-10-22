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
from model_compression_toolkit.core.common.hessian import HessianInfoService, HessianMode, HessianScoresGranularity, \
    HessianScoresRequest
from model_compression_toolkit.core.common.pruning.channels_grouping import ChannelGrouping
from model_compression_toolkit.core.common.pruning.importance_metrics.base_importance_metric import BaseImportanceMetric
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import PruningFrameworkImplementation
from model_compression_toolkit.logger import Logger


class LFHImportanceMetric(BaseImportanceMetric):
    """
    LFHImportanceMetric implements an importance metric based on the Hessian of the
    loss function w.r.t weights of each SIMD group.
    """

    def __init__(self,
                 graph: Graph,
                 representative_data_gen: Callable,
                 fw_impl: PruningFrameworkImplementation,
                 pruning_config: PruningConfig,
                 fw_info: FrameworkInfo):
        """
        Initialize the LFHImportanceMetric instance.

        Args:
            graph (Graph): Computational graph of the model.
            representative_data_gen (Callable): Function to generate representative data.
            fw_impl (PruningFrameworkImplementation): Implementation of pruning for the framework.
            pruning_config (PruningConfig): Configuration for pruning.
            fw_info (FrameworkInfo): Framework-specific information.
        """
        self.float_graph = graph
        self.representative_data_gen = representative_data_gen
        self.fw_impl = fw_impl
        self.pruning_config = pruning_config
        self.fw_info = fw_info

        # Initialize internal dictionaries for storing intermediate computations.
        self._entry_node_to_hessian_score = {}
        self._entry_node_count_oc_nparams = {}
        self._entry_node_to_simd_score = {}

    def get_entry_node_to_simd_score(self, entry_nodes: List[BaseNode]) -> Tuple[Dict[BaseNode, np.ndarray], Dict[BaseNode, List[np.ndarray]]]:
        """
        Compute SIMD scores for each group of channels for a list of entry nodes.
        The function first compute a score for each channel in the node. Then, and based on the scores
        computed, we group the channels into SIMD groups (by simply sorting the scores and grouping them).
        Eventually, we compute a score for each group of channels using LFH score, squared L2-norm,
        and number of parameters in a group.

        Ci_Score = Trace(H_Ci) *  SqL2Norm(Ci) / |Ci|

        Where Trace(H_Ci) is the trace of the hessian of the loss function (w.r.t weights in Ci),
        SqL2Norm is squared l2-norm of the weights in Ci, and |Ci| is the number of parameters in Ci.

        Args:
            entry_nodes (List[BaseNode]): Entry nodes in the graph.

        Returns:
            Tuple[Dict, Dict]: Tuple of dictionaries containing SIMD scores and grouped indices.
        """

        # Compute initial scores for entry nodes.
        entry_node_to_score = self._get_entry_node_to_score(entry_nodes)

        # Group indices based on SIMD configurations.
        grouped_indices = self._compute_simd_groups_indices(entry_node_to_score)

        # Compute squared L2 norms for the groups.
        _squared_l2_norm_by_groups = self._get_squaredl2norm(entry_nodes, grouped_indices)

        # Initialize dictionary for storing SIMD scores.
        entry_node_to_simd_score = {}

        # Compute SIMD scores for each group.
        for node, hessian_score in self._entry_node_to_hessian_score.items():
            group_hessian_score = [np.sum(hessian_score[g]) for g in grouped_indices[node]]
            nparams_by_group = np.asarray([np.sum(self._entry_node_count_oc_nparams[node][g]) for g in grouped_indices[node]])
            entry_node_to_simd_score[node] = np.asarray(group_hessian_score) * _squared_l2_norm_by_groups[node] / nparams_by_group

        return entry_node_to_simd_score, grouped_indices

    def _get_entry_node_to_score(self, entry_nodes: List[BaseNode]) -> Dict[BaseNode, np.ndarray]:
        """
        Compute score for each channel for a list of entry nodes.
        We compute a score for each channel using LFH score, squared L2-norm,
        and number of parameters in the channel.

        Ci_Score = Trace(H_Ci) *  SqL2Norm(Ci) / |Ci|

        Where Trace(H_Ci) is the trace of the hessian of the loss function (w.r.t weights in Ci),
        SqL2Norm is squared l2-norm of the weights in Ci, and |Ci| is the number of parameters in Ci.

        Args:
            entry_nodes (List[BaseNode]): Entry nodes of pruning sections in the graph.

        Returns:
            Dict[BaseNode, np.ndarray]: Dictionary containing channel scores for each entry node.
        """

        # Initialize HessianInfoService for score computation.

        hessian_info_service = HessianInfoService(graph=self.float_graph,
                                                  fw_impl=self.fw_impl)

        # Fetch and process Hessian scores for output channels of entry nodes.
        data_loader = self.fw_impl.convert_data_gen_to_dataloader(self.representative_data_gen, batch_size=1)
        nodes_scores = {}
        for node in entry_nodes:
            request = HessianScoresRequest(mode=HessianMode.WEIGHTS,
                                           granularity=HessianScoresGranularity.PER_OUTPUT_CHANNEL,
                                           target_nodes=[node],
                                           data_loader=data_loader,
                                           n_samples=self.pruning_config.num_score_approximations)
            node_scores = hessian_info_service.fetch_hessian(request)
            nodes_scores.update(node_scores)

        # Average and map scores to nodes.
        self._entry_node_to_hessian_score = {node: np.mean(nodes_scores[node.name], axis=0).flatten() for node in entry_nodes}

        self._entry_node_count_oc_nparams = self._count_oc_nparams(entry_nodes=entry_nodes)
        _entry_node_l2_oc_norm = self._get_squaredl2norm(entry_nodes=entry_nodes)

        # Normalize scores using squared L2 norms and number of parameters.
        _entry_node_to_score = self._normalize_lfh_scores(_entry_node_l2_oc_norm)
        return _entry_node_to_score

    def _compute_simd_groups_indices(self,
                                     entry_node_to_score: Dict[BaseNode, np.ndarray]) -> Dict[BaseNode, List[np.ndarray]]:
        """
        Compute SIMD group indices for each entry node.

        Args:
            entry_node_to_score (Dict[BaseNode, np.ndarray]): Scores for entry nodes.

        Returns:
            Dict[BaseNode, List[np.ndarray]]: Dictionary of entry nodes mapped to their SIMD group indices.
        """
        # Initialize channel grouping utility.
        channel_grouping = ChannelGrouping(prunable_nodes=list(entry_node_to_score.keys()),
                                           fw_info=self.fw_info)

        channel_grouping.group_scores_by_simd_groups(entry_node_to_score)
        grouped_indices = channel_grouping.simd_groups_indices

        return grouped_indices

    def _normalize_lfh_scores(self,
                              entry_node_to_squaredl2norm: Dict[BaseNode, np.ndarray]) -> Dict[BaseNode, np.ndarray]:
        """
        Normalizes the LFH scores using the squared L2 norms.

        Args:
            entry_node_to_squaredl2norm (Dict[BaseNode, np.ndarray]): Squared L2 norms for each entry node.

        Returns:
            Dict[BaseNode, np.ndarray]: Normalized LFH scores for each entry node.
        """
        new_scores = {}
        for node, hessian_score_vector in self._entry_node_to_hessian_score.items():
            # Normalize the hessian score vector using squared L2 norm and the count of output channel parameters.
            new_scores[node] = hessian_score_vector * entry_node_to_squaredl2norm[node] / self._entry_node_count_oc_nparams[node]
        return new_scores

    def _count_oc_nparams(self, entry_nodes: List[BaseNode]) -> Dict[BaseNode, np.ndarray]:
        """
        Counts the number of parameters per output channel for each entry node.

        Args:
            entry_nodes (List[BaseNode]): List of entry nodes to count parameters for.

        Returns:
            Dict[BaseNode, np.ndarray]: Dictionary of nodes and their parameters count per output channel.
        """
        node_channel_params = {}
        for entry_node in entry_nodes:
            kernel_attr, num_oc, oc_axis = self._get_kernel_node_oc_info(entry_node)
            kernel = entry_node.get_weights_by_keys(kernel_attr)

            # Calculate parameters per channel
            params_per_channel = np.prod(kernel.shape) / kernel.shape[oc_axis]
            # Create an array filled with the count of parameters per output channel.
            num_params_array = np.full(kernel.shape[oc_axis], params_per_channel)

            # Map each node to its array of parameters count per output channel.
            node_channel_params[entry_node] = num_params_array
        return node_channel_params

    def _get_squaredl2norm(self,
                           entry_nodes: List[BaseNode],
                           grouped_indices: Dict[BaseNode, List[np.ndarray]] = None) -> Dict[BaseNode, np.ndarray]:
        """
        Computes the squared L2 norm for each output channel (or group of channels) of the entry nodes.

        Args:
            entry_nodes (List[BaseNode]): List of entry nodes for L2 norm computation.
            grouped_indices (Dict[BaseNode, List[List[int]]], optional): Indices of channel groups. Defaults to None.

        Returns:
            Dict[BaseNode, np.ndarray]: Dictionary of nodes and their squared L2 norms for each output channel (or group).
        """
        node_l2_channel_norm = {}
        for entry_node in entry_nodes:
            kernel_attr, num_oc, oc_axis = self._get_kernel_node_oc_info(entry_node)
            # Retrieve the kernel tensor of the node.
            kernel = entry_node.get_weights_by_keys(kernel_attr)
            # Split the kernel tensor into individual channels (or groups if provided).
            channels = np.split(kernel, indices_or_sections=num_oc, axis=oc_axis)

            # If grouped_indices are provided, concatenate tensors based on grouped indices.
            if grouped_indices:
                concatenated_tensors = self._concatenate_tensors_by_indices(channels, grouped_indices[entry_node])
                channels = concatenated_tensors

            # Compute the squared L2 norm for each channel (or group).
            l2_norms = np.asarray([np.linalg.norm(c.flatten(), ord=2) ** 2 for c in channels])
            node_l2_channel_norm[entry_node] = l2_norms

        return node_l2_channel_norm

    def _get_kernel_node_oc_info(self, entry_node: BaseNode) -> Tuple[str, int, int]:
        """
        Retrieves information about the output channels (oc) for a given kernel node.

        Args:
            entry_node (BaseNode): The node whose output channel information is needed.

        Returns:
            tuple: A tuple containing the kernel attribute, the number of output channels, and the axis of the output channels.
        """
        kernel_attr = self.fw_info.get_kernel_op_attributes(entry_node.type)
        # Ensure only one kernel attribute exists for the given node.
        if len(kernel_attr) != 1:
            Logger.critical(f"Expected a single attribute but found multiple attributes ({len(kernel_attr)}) for node {entry_node}.")
        kernel_attr = kernel_attr[0]

        # Retrieve and validate the axis index for the output channels.
        oc_axis, _ = self.fw_info.kernel_channels_mapping.get(entry_node.type)
        if oc_axis is None or int(oc_axis) != oc_axis:
            Logger.critical(f"Invalid output channel axis type for node {entry_node}: expected integer but got {oc_axis}.")

        # Get the number of output channels based on the kernel attribute and axis.
        num_oc = entry_node.get_weights_by_keys(kernel_attr[0]).shape[oc_axis]
        return kernel_attr, num_oc, oc_axis

    def _concatenate_tensors_by_indices(self,
                                        channels: List[np.ndarray],
                                        index_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Concatenates tensors based on provided indices.

        Args:
            channels (List[np.ndarray]): List of channel tensors.
            index_list (List[np.ndarray]): Indices of channels to be concatenated.

        Returns:
            List[np.ndarray]: List of concatenated tensors.
        """
        concatenated_tensors = []
        for index_array in index_list:
            # Gather tensors based on indices.
            tensors_to_concatenate = [channels[i] for i in index_array]
            # Concatenate the gathered tensors.
            concatenated_tensor = np.concatenate(tensors_to_concatenate)
            concatenated_tensors.append(concatenated_tensor)
        return concatenated_tensors
