# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Callable, Tuple
from typing import Dict, List
import numpy as np

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode, \
    VirtualSplitWeightsNode, VirtualSplitActivationNode
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import RUTarget, ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.ru_aggregation_methods import MpRuAggregation
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.ru_methods import MpRuMetric
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import SensitivityEvaluation


class MixedPrecisionSearchManager:
    """
    Class to wrap and manage the search process of a mixed-precision configuration.
    """

    def __init__(self,
                 graph: Graph,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation,
                 sensitivity_evaluator: SensitivityEvaluation,
                 ru_functions: Dict[RUTarget, Tuple[MpRuMetric, MpRuAggregation]],
                 target_resource_utilization: ResourceUtilization,
                 original_graph: Graph = None):
        """

        Args:
            graph: Graph to search for its MP configuration.
            fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
            fw_impl: FrameworkImplementation object with specific framework methods implementation.
            sensitivity_evaluator: A SensitivityEvaluation which provides a function that evaluates the sensitivity of
                a bit-width configuration for the MP model.
            ru_functions: A dictionary with pairs of (MpRuMethod, MpRuAggregationMethod) mapping a RUTarget to
                a couple of resource utilization metric function and resource utilization aggregation function.
            target_resource_utilization: Target Resource Utilization to bound our feasible solution space s.t the configuration does not violate it.
            original_graph: In case we have a search over a virtual graph (if we have BOPS utilization target), then this argument
                will contain the original graph (for config reconstruction purposes).
        """

        self.graph = graph
        self.original_graph = graph if original_graph is None else original_graph
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.sensitivity_evaluator = sensitivity_evaluator
        self.layer_to_bitwidth_mapping = self.get_search_space()
        self.compute_metric_fn = self.get_sensitivity_metric()

        self.compute_ru_functions = ru_functions
        self.target_resource_utilization = target_resource_utilization
        self.min_ru_config = self.graph.get_min_candidates_config(fw_info)
        self.max_ru_config = self.graph.get_max_candidates_config(fw_info)
        self.min_ru = self.compute_min_ru()
        self.non_conf_ru_dict = self._non_configurable_nodes_ru()

        self.config_reconstruction_helper = ConfigReconstructionHelper(virtual_graph=self.graph,
                                                                       original_graph=self.original_graph)

    def get_search_space(self) -> Dict[int, List[int]]:
        """
        The search space is a mapping from a node's index to a list of integers (possible bitwidths candidates indeces
        for the node).

        Returns:
            The entire search space of the graph.
        """

        indices_mapping = {}
        nodes_to_configure = self.graph.get_configurable_sorted_nodes(self.fw_info)
        for idx, n in enumerate(nodes_to_configure):
            # For each node, get all possible bitwidth indices for it
            # (which is a list from 0 to the length of the candidates mp_config list of the node).
            indices_mapping[idx] = list(range(len(n.candidates_quantization_cfg)))  # all search_methods space
        return indices_mapping

    def get_sensitivity_metric(self) -> Callable:
        """

        Returns: Return a function (from the framework implementation) to compute a metric that
        indicates the similarity of the mixed-precision model (to the float model) for a given
        mixed-precision configuration.

        """
        # Get from the framework an evaluation function on how a MP configuration,
        # affects the expected loss.

        return self.sensitivity_evaluator.compute_metric

    def compute_min_ru(self) -> Dict[RUTarget, np.ndarray]:
        """
        Computes a resource utilization vector with the values matching to the minimal mp configuration
        (i.e., each node is configured with the quantization candidate that would give the minimal size of the
        node's resource utilization).
        The method computes the minimal resource utilization vector for each target resource utilization.

        Returns: A dictionary mapping each target resource utilization to its respective minimal 
        resource utilization values.

        """
        min_ru = {}
        for ru_target, ru_fns in self.compute_ru_functions.items():
            # ru_fns is a pair of resource utilization computation method and 
            # resource utilization aggregation method (in this method we only need the first one)
            min_ru[ru_target] = ru_fns[0](self.min_ru_config, self.graph, self.fw_info, self.fw_impl)

        return min_ru

    def compute_resource_utilization_matrix(self, target: RUTarget) -> np.ndarray:
        """
        Computes and builds a resource utilization matrix, to be used for the mixed-precision search problem formalization.
        The matrix is constructed as follows (for a given target):
        - Each row represents the set of resource utilization values for a specific resource utilization 
            measure (number of rows should be equal to the length of the output of the respective target compute_ru function).
        - Each entry in a specific column represents the resource utilization value of a given configuration 
            (single layer is configured with specific candidate, all other layer are at the minimal resource 
            utilization configuration) for the resource utilization measure of the respective row.

        Args:
            target: The resource target for which the resource utilization is calculated (a RUTarget value).

        Returns: A resource utilization matrix.

        """
        assert isinstance(target, RUTarget), f"{target} is not a valid resource target"

        configurable_sorted_nodes = self.graph.get_configurable_sorted_nodes(self.fw_info)

        ru_matrix = []
        for c, c_n in enumerate(configurable_sorted_nodes):
            for candidate_idx in range(len(c_n.candidates_quantization_cfg)):
                if candidate_idx == self.min_ru_config[c]:
                    # skip ru computation for min configuration. Since we compute the difference from min_ru it'll
                    # always be 0 for all entries in the results vector.
                    candidate_rus = np.zeros(shape=self.min_ru[target].shape)
                else:
                    candidate_rus = self.compute_candidate_relative_ru(c, candidate_idx, target)
                ru_matrix.append(np.asarray(candidate_rus))

        # We need to transpose the calculated ru matrix to allow later multiplication with
        # the indicators' diagonal matrix.
        # We only move the first axis (num of configurations) to be last,
        # the remaining axes include the metric specific nodes (rows dimension of the new tensor)
        # and the ru metric values (if they are non-scalars)
        np_ru_matrix = np.array(ru_matrix)
        return np.moveaxis(np_ru_matrix, source=0, destination=len(np_ru_matrix.shape) - 1)

    def compute_candidate_relative_ru(self,
                                      conf_node_idx: int,
                                      candidate_idx: int,
                                      target: RUTarget) -> np.ndarray:
        """
        Computes a resource utilization vector for a given candidates of a given configurable node, 
        i.e., the matching resource utilization vector which is obtained by computing the given target's 
        resource utilization function on a minimal configuration in which the given
        layer's candidates is changed to the new given one.
        The result is normalized by subtracting the target's minimal resource utilization vector.

        Args:
            conf_node_idx: The index of a node in a sorted configurable nodes list.
            candidate_idx: The index of a node's quantization configuration candidate.
            target: The target for which the resource utilization is calculated (a RUTarget value).

        Returns: Normalized node's resource utilization vector

        """
        return self.compute_node_ru_for_candidate(conf_node_idx, candidate_idx, target) - \
               self.get_min_target_resource_utilization(target)

    def get_min_target_resource_utilization(self, target: RUTarget) -> np.ndarray:
        """
        Returns the minimal resource utilization vector (pre-calculated on initialization) of a specific target.

        Args:
            target: The target for which the resource utilization is calculated (a RUTarget value).

        Returns: Minimal resource utilization vector.

        """
        return self.min_ru[target]

    def compute_node_ru_for_candidate(self, conf_node_idx: int, candidate_idx: int, target: RUTarget) -> np.ndarray:
        """
        Computes a resource utilization vector after replacing the given node's configuration candidate in the minimal
        target configuration with the given candidate index.

        Args:
            conf_node_idx: The index of a node in a sorted configurable nodes list.
            candidate_idx: Quantization config candidate to be used for the node's resource utilization computation.
            target: The target for which the resource utilization is calculated (a RUTarget value).

        Returns: Node's resource utilization vector.

        """
        return self.compute_ru_functions[target][0](
            self.replace_config_in_index(
                self.min_ru_config,
                conf_node_idx,
                candidate_idx),
            self.graph,
            self.fw_info,
            self.fw_impl)

    @staticmethod
    def replace_config_in_index(mp_cfg: List[int], idx: int, value: int) -> List[int]:
        """
        Replacing the quantization configuration candidate in a given mixed-precision configuration at the given
        index (node's index) with the given value (candidate index).

        Args:
            mp_cfg: Mixed-precision configuration (list of candidates' indices)
            idx: A configurable node's index.
            value: A new candidate index to configure.

        Returns: A new mixed-precision configuration.

        """
        updated_cfg = mp_cfg.copy()
        updated_cfg[idx] = value
        return updated_cfg

    def _non_configurable_nodes_ru(self) -> Dict[RUTarget, np.ndarray]:
        """
        Computes a resource utilization vector of all non-configurable nodes in the given graph for each of the 
        resource utilization targets.

        Returns: A mapping between a RUTarget and its non-configurable nodes' resource utilization vector.
        """

        non_conf_ru_dict = {}
        for target, ru_value in self.target_resource_utilization.get_resource_utilization_dict().items():
            # Call for the ru method of the given target - empty quantization configuration list is passed since we
            # compute for non-configurable nodes
            if target == RUTarget.BOPS:
                ru_vector = None
            else:
                ru_vector = self.compute_ru_functions[target][0]([], self.graph, self.fw_info, self.fw_impl)

            non_conf_ru_dict[target] = ru_vector

        return non_conf_ru_dict

    def compute_resource_utilization_for_config(self, config: List[int]) -> ResourceUtilization:
        """
        Computes the resource utilization values for a given mixed-precision configuration.

        Args:
            config: A mixed-precision configuration (list of candidates indices)

        Returns: A ResourceUtilization object with the model's resource utilization values when quantized 
        with the given config.

        """

        ru_dict = {}

        for ru_target, ru_fns in self.compute_ru_functions.items():
            # Passing False to ru methods and aggregations to indicates that the computations
            # are not for constraints setting
            if ru_target == RUTarget.BOPS:
                configurable_nodes_ru_vector = ru_fns[0](config, self.original_graph, self.fw_info, self.fw_impl, False)
            else:
                configurable_nodes_ru_vector = ru_fns[0](config, self.original_graph, self.fw_info, self.fw_impl)
            non_configurable_nodes_ru_vector = self.non_conf_ru_dict.get(ru_target)
            if non_configurable_nodes_ru_vector is None or len(non_configurable_nodes_ru_vector) == 0:
                ru_ru = self.compute_ru_functions[ru_target][1](configurable_nodes_ru_vector, False)
            else:
                ru_ru = self.compute_ru_functions[ru_target][1](
                    np.concatenate([configurable_nodes_ru_vector, non_configurable_nodes_ru_vector]), False)

            ru_dict[ru_target] = ru_ru[0]

        config_ru = ResourceUtilization()
        config_ru.set_resource_utilization_by_target(ru_dict)
        return config_ru

    def finalize_distance_metric(self, layer_to_metrics_mapping: Dict[int, Dict[int, float]]):
        """
        Finalizing the distance metric building.
        The method checks to see if the maximal distance value is larger than a given threshold, and if so,
        it scales all metric values to prevent possible numerical issues.
        Modification to the dictionary is done inplace.

        Args:
            layer_to_metrics_mapping: A mapping between a node index to a mapping between
            a bitwidth index to a distance value.

        """
        # normalize metric for numerical stability

        max_dist = max([max([d for b, d in dists.items()]) for layer, dists in layer_to_metrics_mapping.items()])
        if max_dist >= self.sensitivity_evaluator.quant_config.metric_normalization_threshold:
            Logger.warning(f"The mixed precision distance metric values indicate a large error in the quantized model."
                           f"this can cause numerical issues."
                           f"The program will proceed with mixed precision search after scaling the metric values,"
                           f"which can lead to unstable results.")
            for layer, dists in layer_to_metrics_mapping.items():
                for b, d in dists.items():
                    layer_to_metrics_mapping[layer][b] /= max_dist


class ConfigReconstructionHelper:
    """
    A class to help reconstruct an original mixed-precision configuration from a virtual one,
    when running mixed-precision search with BOPS utilization.
    It provides a reconstruct_config_from_virtual_graph which allows to translate a bit-width config of a virtual graph
    to a config of the original configurable nodes.
    """

    def __init__(self, virtual_graph: Graph, original_graph: Graph):
        """
        Init a ConfigReconstructionHelper object.
        It holds a dictionary variable named origin_node_idx_to_cfg which holds the mapping from an original graph's
        configurable node to its actual bit-width index (this data structure is being cleared
        after every reconstruction call).

        Args:
            virtual_graph: The virtual graph.
            original_graph: The original graph.
        """

        self.virtual_graph = virtual_graph
        self.original_graph = original_graph
        self.fw_info = original_graph.fw_info

        self.virtual_sorted_nodes_names = self.virtual_graph.get_configurable_sorted_nodes_names(self.fw_info)
        self.origin_sorted_conf_nodes_names = self.original_graph.get_configurable_sorted_nodes_names(self.fw_info)

        self.origin_node_idx_to_cfg = {}

    def _clear_reconstruction_dict(self):
        """
        Clears the origin_node_idx_to_cfg data structure.
        """

        self.origin_node_idx_to_cfg = {}

    def reconstruct_config_from_virtual_graph(self,
                                              virtual_mp_cfg: List[int],
                                              changed_virtual_nodes_idx: List[int] = None,
                                              original_base_config: List[int] = None) -> List[int]:
        """
        Reconstructs the original config for a given virtual graph mixed-precision config.
        It iterates over all virtual configurable node (that has some chosen bit-width virtual candidate)
        and translates its chosen candidate to a candidate index of configurable nodes in the original graph.
        The translation is based of the virtual node's type. Note that if the node is a split activation node
        for instance, then we need to find its matching weights node in order to construct the original linear node's
        chosen config.

        Args:
            virtual_mp_cfg: A mixed-precision configuration (list of candidates indices) of the virtual graph.
            changed_virtual_nodes_idx: Provide an optional list of virtual nodes indices for which the
                config reconstruction will be computed.
            original_base_config: If changed_virtual_nodes_idx is provided, need to provide a base config from which the
                bit-width for all un-changed original nodes will be taken.

        Returns: A mixed-precision configuration (list of candidates indices) of the original graph.

        """

        if changed_virtual_nodes_idx is not None:
            if original_base_config is None:
                Logger.critical("To run config reconstruction for a partial set of nodes, a base original config must be provided.")  # pragma: no cover

            updated_virtual_nodes = \
                [(idx, self.virtual_graph.get_configurable_sorted_nodes(self.fw_info)[idx]) for idx in changed_virtual_nodes_idx]
            # Iterating only over the virtual nodes that have updated config
            for virtual_node_idx, n in updated_virtual_nodes:
                self.reconstruct_node_config(n, virtual_mp_cfg, virtual_node_idx)
            # Updating reconstructed config for all other nodes based on provided base_config
            original_sorted_conf_nodes = self.original_graph.get_configurable_sorted_nodes(self.fw_info)
            for i in range(len(original_base_config)):
                if i not in list(self.origin_node_idx_to_cfg.keys()):
                    self.update_config_at_original_idx(n=original_sorted_conf_nodes[i],
                                                       origin_cfg_idx=original_base_config[i])
        else:
            # Reconstruct entire config
            for virtual_node_idx, n in enumerate(self.virtual_graph.get_configurable_sorted_nodes(self.fw_info)):
                self.reconstruct_node_config(n, virtual_mp_cfg, virtual_node_idx)

        res_config = [self.origin_node_idx_to_cfg[key] for key in sorted(self.origin_node_idx_to_cfg.keys())]
        self._clear_reconstruction_dict()
        return res_config

    def reconstruct_node_config(self,
                                n: BaseNode,
                                virtual_mp_cfg: List[int],
                                virtual_node_idx: int):
        """
        Reconstructs the original configuration for a single node. Updates the mapping inplace.

        Args:
            n: The node to reconstruct the configuration for.
            virtual_mp_cfg: A mixed-precision configuration (list of candidates indices) of the virtual graph.
            virtual_node_idx: The index of the virtual node in the virtual mixed-precision configuration.
        """

        virtual_cfg_idx = virtual_mp_cfg[virtual_node_idx]

        if isinstance(n, VirtualActivationWeightsNode):
            weights_node = n.original_weights_node
            if isinstance(weights_node, VirtualSplitWeightsNode):
                self.get_activation_for_split_weights(weights_node, n, virtual_cfg_idx, virtual_mp_cfg)
            else:
                Logger.critical(f"Virtual graph construction error: Expected all weights nodes to be split into weights and activation nodes. Found node '{n.name}' not split as expected. Every weights node should correspond to a VirtualSplitWeightsNode type.")  # pragma: no cover

            activation_node = n.original_activation_node
            if isinstance(activation_node, VirtualSplitActivationNode):
                self.get_weights_for_split_activation(activation_node, n, virtual_cfg_idx, virtual_mp_cfg)
            else:
                if activation_node.name in self.origin_sorted_conf_nodes_names:
                    # It is possible that the original activation node is not configurable,
                    # in this case we don't need to retrieve its bit-width config
                    self.retrieve_activation_only_config(activation_node, n, virtual_cfg_idx)
        elif isinstance(n, VirtualSplitWeightsNode):
            # If the node's predecessor have multiple outgoing edges then it is possible that this weights
            # node is not composed with an activation, but otherwise there is something wrong, and we need
            # to raise an exception
            predecessor = self.virtual_graph.get_prev_nodes(n)
            assert len(predecessor) == 1  # Sanity check
            predecessor = predecessor[0]
            if len(self.virtual_graph.out_edges(predecessor)) > 1:
                # It's ok, need to find the node's configuration
                self.get_activation_for_split_weights(n, n, virtual_cfg_idx, virtual_mp_cfg)
            else:
                Logger.critical(f"Virtual graph configuration error: Expected the predecessor of node '{n.name}' to have multiple outputs when not composed with an activation node.")  # pragma: no cover
        elif isinstance(n, VirtualSplitActivationNode):
            self.get_weights_for_split_activation(n, n, virtual_cfg_idx, virtual_mp_cfg)
        else:
            # Node didn't change in virtual graph - candidates list is similar to original
            if n.name not in self.origin_sorted_conf_nodes_names:
                Logger.critical(f"Configuration mismatch: Node '{n.name}' is configurable in the virtual graph but not in the original graph. Verify node configurations.")  # pragma: no cover
            origin_idx = self.origin_sorted_conf_nodes_names.index(n.name)
            self.origin_node_idx_to_cfg[origin_idx] = virtual_cfg_idx

    def retrieve_weights_only_config(self, weights_node: BaseNode, virtual_node: BaseNode, virtual_cfg_idx: int):
        """
        Retrieves the configuration of an original weights configurable node based on a
        virtual weights configurable node's chosen config idx, and updates (inplace) the origin_cfg_idx mapping dict.
        If the original node is not configurable, nothing will be updated.

        Args:
            weights_node: The original weights (possibly configurable) node.
            virtual_node: The virtual weights configurable node.
            virtual_cfg_idx: The virtual node's chosen config index.
        """

        if weights_node.name in self.origin_sorted_conf_nodes_names:
            # It is possible that the original weights node is not configurable,
            # in this case we don't need to retrieve its bit-width config
            kernel_attr = self.fw_info.get_kernel_op_attributes(weights_node.type)[0]
            weights_bitwidth = (virtual_node.candidates_quantization_cfg[virtual_cfg_idx].weights_quantization_cfg
                                .get_attr_config(kernel_attr).weights_n_bits)
            origin_cfg_idx = [i for i, c in
                              enumerate(weights_node.candidates_quantization_cfg) if
                              c.weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits == weights_bitwidth]

            self.update_config_at_original_idx(weights_node, origin_cfg_idx[0])

    def retrieve_activation_only_config(self, activation_node: BaseNode, virtual_node: BaseNode, virtual_cfg_idx: int):
        """
        Retrieves the configuration of an original activation configurable node based on a
        virtual activation configurable node's chosen config idx, and updates (inplace) the origin_cfg_idx mapping dict.
        If the original node is not configurable, nothing will be updated.

        Args:
            activation_node: The original activation (possibly configurable) node.
            virtual_node: The virtual activation configurable node.
            virtual_cfg_idx: The virtual node's chosen config index.
        """

        if activation_node.name in self.origin_sorted_conf_nodes_names:
            # It is possible that the original activation node is not configurable,
            # in this case we don't need to retrieve its bit-width config
            activation_bitwidth = virtual_node.candidates_quantization_cfg[
                virtual_cfg_idx].activation_quantization_cfg.activation_n_bits
            origin_cfg_idx = [i for i, c in
                              enumerate(activation_node.candidates_quantization_cfg) if
                              c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]

            self.update_config_at_original_idx(activation_node, origin_cfg_idx[0])

    def retrieve_activation_weights_config(self,
                                           activation_node: BaseNode,
                                           weights_node: BaseNode,
                                           virtual_node: BaseNode,
                                           virtual_cfg_idx: int,
                                           virtual_mp_cfg: List[int]):
        """
        Retrieves the configuration of an original weights and activation (possibly) configurable node based on a given
        virtual split weights node and a virtual split activation node which represents its matching in the original graph.
        it updates (inplace) the origin_cfg_idx mapping dict.

        Args:
            activation_node: The virtual node that contains the activation that matches the weights node in the original graph.
            weights_node: The virtual node that contains the weights representation of an original node.
            virtual_node: The virtual node that contains the virtual weights node (either a composed node or a split weights node).
            virtual_cfg_idx: The virtual node's chosen config index.
            virtual_mp_cfg: The virtual graph's chosen mp config.
        """

        activation_bitwidth = activation_node.candidates_quantization_cfg[virtual_mp_cfg[
            self.virtual_sorted_nodes_names.index(activation_node.name)]].activation_quantization_cfg.activation_n_bits

        kernel_attr = self.fw_info.get_kernel_op_attributes(weights_node.type)[0]

        weights_bitwidth = (virtual_node.candidates_quantization_cfg[virtual_cfg_idx].weights_quantization_cfg
                            .get_attr_config(kernel_attr).weights_n_bits)

        origin_cfg_idx = [i for i, c in
                          enumerate(weights_node.origin_node.candidates_quantization_cfg) if
                          c.weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits == weights_bitwidth and
                          c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]

        self.update_config_at_original_idx(weights_node.origin_node, origin_cfg_idx[0])

    def retrieve_weights_activation_config(self,
                                           activation_node: BaseNode,
                                           weights_node: BaseNode,
                                           virtual_node: BaseNode,
                                           virtual_cfg_idx: int,
                                           virtual_mp_cfg: List[int]):
        """
        Retrieves the configuration of an original weights and activation (possibly) configurable node based on a given
        virtual split activation node and a virtual split weights node which represents its matching in the original graph.
        it updates (inplace) the origin_cfg_idx mapping dict.

        Args:
            activation_node: The virtual node that contains the activation representation of an original node.
            weights_node: The virtual node that contains the weights that matches the activation node in the original graph.
            virtual_node: The virtual node that contains the virtual activation node (either a composed node or a split activation node).
            virtual_cfg_idx: The virtual node's chosen config index.
            virtual_mp_cfg: The virtual graph's chosen mp config.
        """

        kernel_attr = self.fw_info.get_kernel_op_attributes(weights_node.type)[0]

        weights_bitwidth = (weights_node.candidates_quantization_cfg[virtual_mp_cfg[
            self.virtual_sorted_nodes_names.index(weights_node.name)]]
                            .weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits)

        activation_bitwidth = virtual_node.candidates_quantization_cfg[
            virtual_cfg_idx].activation_quantization_cfg.activation_n_bits

        origin_cfg_idx = [i for i, c in enumerate(activation_node.origin_node.candidates_quantization_cfg) if
                          c.weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits == weights_bitwidth and
                          c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]

        self.update_config_at_original_idx(activation_node.origin_node, origin_cfg_idx[0])

    def get_activation_for_split_weights(self,
                                         weights_node: BaseNode,
                                         virtual_node: BaseNode,
                                         virtual_cfg_idx: int,
                                         virtual_mp_cfg: List[int]):
        """
        Finds the matching activation node in the virtual graph for a given split weights node,
        and calls the relevant method for updating the configuration mapping.

        Args:
            weights_node: A virtual weights node.
            virtual_node: A virtual node that contains the virtual weights node (either a composed node or a split weights node).
            virtual_cfg_idx: The virtual node's chosen config index.
            virtual_mp_cfg: The virtual graph's chosen mp config.

        """

        # This is a weights node that was split, means it has an activation node that should follow it,
        # and we need its configuration in order to reconstruct the original node's configuration.
        matching_activation_node = self.virtual_graph.get_next_nodes(virtual_node)
        assert len(matching_activation_node) == 1
        activation_node = matching_activation_node[0]

        if isinstance(activation_node, VirtualActivationWeightsNode):
            if activation_node.original_activation_node.is_activation_quantization_enabled() and not \
                    activation_node.original_activation_node.is_all_activation_candidates_equal():
                assert activation_node.name in self.virtual_sorted_nodes_names  # Sanity check
                # The original node is both weights and activation configurable
                self.retrieve_activation_weights_config(activation_node, weights_node, virtual_node, virtual_cfg_idx, virtual_mp_cfg)
            else:
                # weights_node here is a split weights node therefore must have 'origin_node'
                self.retrieve_weights_only_config(weights_node.origin_node, virtual_node, virtual_cfg_idx)
        else:
            assert isinstance(activation_node, VirtualSplitActivationNode)  # Sanity check
            if activation_node.name in self.virtual_sorted_nodes_names:
                self.retrieve_activation_weights_config(activation_node, weights_node, virtual_node, virtual_cfg_idx, virtual_mp_cfg)
            else:
                # The original node is only weights configurable
                # weights_node here is a split weights node therefore must have 'origin_node'
                self.retrieve_weights_only_config(weights_node.origin_node, virtual_node, virtual_cfg_idx)

    def get_weights_for_split_activation(self,
                                         activation_node: BaseNode,
                                         virtual_node: BaseNode,
                                         virtual_cfg_idx: int,
                                         virtual_mp_cfg: List[int]):
        """
        Finds the matching weights node in the virtual graph for a given split activation node,
        and calls the relevant method for updating the configuration mapping.

        Args:
            activation_node: A virtual activation node.
            virtual_node: A virtual node that contains the virtual activation node (either a composed node or a split activation node).
            virtual_cfg_idx: The virtual node's chosen config index.
            virtual_mp_cfg: The virtual graph's chosen mp config.

        """

        # This is an activation node that was split, means it has a weights node that should come before it,
        # and we need its configuration in order to reconstruct the original node's configuration.
        matching_weights_node = self.virtual_graph.get_prev_nodes(virtual_node)
        assert len(matching_weights_node) == 1
        weights_node = matching_weights_node[0]

        if isinstance(weights_node, VirtualActivationWeightsNode):
            kernel_attr = self.fw_info.get_kernel_op_attributes(weights_node.type)[0]
            if weights_node.original_weights_node.is_weights_quantization_enabled(kernel_attr) and not \
                    weights_node.original_weights_node.is_all_weights_candidates_equal(kernel_attr):
                assert weights_node.name in self.virtual_sorted_nodes_names  # Sanity check
                # The original node is both weights and activation configurable
                self.retrieve_weights_activation_config(activation_node, weights_node, virtual_node, virtual_cfg_idx, virtual_mp_cfg)
            else:
                # The original node is only activation configurable
                # activation_node here is a split activation node therefore must have 'origin_node'
                self.retrieve_activation_only_config(activation_node.origin_node, virtual_node, virtual_cfg_idx)
        else:
            # If the node's predecessor e multiple outgoing edges than it is possible that this weights
            # node is not composed with an activation, but otherwise this is something wrong and we need
            # to raise an exception
            predecessor = self.virtual_graph.get_prev_nodes(weights_node)
            assert len(predecessor) == 1  # Sanity check
            predecessor = predecessor[0]
            if len(self.virtual_graph.out_edges(predecessor)) > 1:
                # It's ok, need to find the node's configuration
                self.retrieve_weights_activation_config(activation_node, weights_node, virtual_node, virtual_cfg_idx, virtual_mp_cfg)
            else:
                Logger.critical(f"Virtual graph configuration error: Expected the predecessor of node '{n.name}' to have multiple outputs when not composed with an activation node.")  # pragma: no cover

    def update_config_at_original_idx(self, n: BaseNode, origin_cfg_idx: int):
        """
        Updates (inplace) the origin_node_idx_to_cfg mapping wit hthe given index for a given original node index
        (in the original graph's sorted configurable nodes list).

        Args:
            n: An original graph's node
            origin_cfg_idx: A candidate index.

        """

        origin_idx = self.origin_sorted_conf_nodes_names.index(n.name)
        self.origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx
