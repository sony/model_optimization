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
import copy
from collections import defaultdict

from tqdm import tqdm

from typing import Dict, List, Tuple

import numpy as np

from model_compression_toolkit.constants import EPS
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode, \
    VirtualSplitWeightsNode, VirtualSplitActivationNode
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget, ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    TargetInclusionCriterion, BitwidthMode
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_ru_helper import \
    MixedPrecisionRUHelper
from model_compression_toolkit.core.common.mixed_precision.search_methods.linear_programming import \
    MixedPrecisionIntegerLPSolver
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import SensitivityEvaluation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.logger import Logger


class MixedPrecisionSearchManager:
    """
    Class to wrap and manage the search process of a mixed-precision configuration.
    """

    def __init__(self,
                 graph: Graph,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation,
                 sensitivity_evaluator: SensitivityEvaluation,
                 target_resource_utilization: ResourceUtilization):
        """

        Args:
            graph: Graph to search for its MP configuration.
            fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
            fw_impl: FrameworkImplementation object with specific framework methods implementation.
            sensitivity_evaluator: A SensitivityEvaluation which provides a function that evaluates the sensitivity of
                a bit-width configuration for the MP model.
            target_resource_utilization: Target Resource Utilization to bound our feasible solution space s.t the configuration does not violate it.
        """

        self.fw_info = fw_info
        self.fw_impl = fw_impl

        self.original_graph = graph
        # graph for mp search
        self.mp_graph, self.using_virtual_graph = self._get_mp_graph(graph, target_resource_utilization)
        del graph  # so that it's not used by mistake

        self.sensitivity_evaluator = sensitivity_evaluator
        self.target_resource_utilization = target_resource_utilization

        self.mp_topo_configurable_nodes = self.mp_graph.get_configurable_sorted_nodes(fw_info)
        self.layer_to_bitwidth_mapping = self.get_search_space()

        self.ru_targets = target_resource_utilization.get_restricted_targets()
        self.ru_helper = MixedPrecisionRUHelper(self.mp_graph, fw_info, fw_impl)

        self.min_ru_config = self.mp_graph.get_min_candidates_config(fw_info)
        self.max_ru_config = self.mp_graph.get_max_candidates_config(fw_info)
        self.min_ru = self.ru_helper.compute_utilization(self.ru_targets, self.min_ru_config)

        self.config_reconstruction_helper = ConfigReconstructionHelper(virtual_graph=self.mp_graph,
                                                                       original_graph=self.original_graph)

    def search(self) -> List[int]:
        """
        Run mixed precision search.

        Returns:
            Indices of the selected bit-widths candidates.
        """
        candidates_sensitivity = self._build_sensitivity_mapping()
        candidates_ru = self._compute_relative_ru_matrices()
        rel_target_ru = self._get_relative_ru_constraint_per_mem_element()
        solver = MixedPrecisionIntegerLPSolver(candidates_sensitivity, candidates_ru, rel_target_ru)
        config = solver.run()

        if self.using_virtual_graph:
            config = self.config_reconstruction_helper.reconstruct_config_from_virtual_graph(config)
        return config

    def _get_relative_ru_constraint_per_mem_element(self) -> Dict[RUTarget, np.ndarray]:
        """
        Computes resource utilization constraint with respect to the minimal bit configuration, i.e. corresponding
        constraint for each memory element is the relative utilization between the target utilization and
        element's utilization for min-bit configuration.

        Returns:
            A dictionary of relative resource utilization constraints per ru target.

        Raises:
            ValueError: if target resource utilization cannot be satisfied (utilization for the minimal bit
              configuration exceeds the requested target utilization for any target).
        """
        target_ru = self.target_resource_utilization.get_resource_utilization_dict(restricted_only=True)
        rel_target_ru = {
            ru_target: ru - self.min_ru[ru_target] for ru_target, ru in target_ru.items()
        }
        unsatisfiable_targets = {
            ru_target.value: target_ru[ru_target] for ru_target, ru in rel_target_ru.items() if any(ru < 0)
        }
        if unsatisfiable_targets:
            raise ValueError(f"The model cannot be quantized to meet the specified resource utilization for the "
                             f"following targets: {unsatisfiable_targets}")
        return rel_target_ru

    def _build_sensitivity_mapping(self, eps: float = EPS) -> Dict[int, Dict[int, float]]:
        """
        This function measures the sensitivity of a change in a bitwidth of a layer on the entire model.
        It builds a mapping from a node's index, to its bitwidht's effect on the model sensitivity.
        For each node and some possible node's bitwidth (according to the given search space), we use
        the framework function compute_metric_fn in order to infer
        a batch of images, and compute (using the inference results) the sensitivity metric of
        the configured mixed-precision model.

        Args:
            eps: Epsilon value to manually increase metric value (if necessary) for numerical stability

        Returns:
            Mapping from each node's index in a graph, to a dictionary from the bitwidth index (of this node) to
            the sensitivity of the model.

        """

        Logger.info('Starting to evaluate metrics')
        layer_to_metrics_mapping = {}

        compute_metric = self.sensitivity_evaluator.compute_metric
        if self.using_virtual_graph:
            origin_max_config = self.config_reconstruction_helper.reconstruct_config_from_virtual_graph(
                self.max_ru_config)
            max_config_value = compute_metric(origin_max_config)
        else:
            max_config_value = compute_metric(self.max_ru_config)

        for node_idx, layer_possible_bitwidths_indices in tqdm(self.layer_to_bitwidth_mapping.items(),
                                                               total=len(self.layer_to_bitwidth_mapping)):
            layer_to_metrics_mapping[node_idx] = {}

            for bitwidth_idx in layer_possible_bitwidths_indices:
                if self.max_ru_config[node_idx] == bitwidth_idx:
                    # This is a computation of the metric for the max configuration, assign pre-calculated value
                    layer_to_metrics_mapping[node_idx][bitwidth_idx] = max_config_value
                    continue

                # Create a configuration that differs at one layer only from the baseline model
                mp_model_configuration = self.max_ru_config.copy()
                mp_model_configuration[node_idx] = bitwidth_idx

                # Build a distance matrix using the function we got from the framework implementation.
                if self.using_virtual_graph:
                    # Reconstructing original graph's configuration from virtual graph's configuration
                    origin_mp_model_configuration = \
                        self.config_reconstruction_helper.reconstruct_config_from_virtual_graph(
                            mp_model_configuration,
                            changed_virtual_nodes_idx=[node_idx],
                            original_base_config=origin_max_config)
                    origin_changed_nodes_indices = [i for i, c in enumerate(origin_max_config) if
                                                    c != origin_mp_model_configuration[i]]
                    metric_value = compute_metric(
                        origin_mp_model_configuration,
                        origin_changed_nodes_indices,
                        origin_max_config)
                else:
                    metric_value = compute_metric(
                        mp_model_configuration,
                        [node_idx],
                        self.max_ru_config)

                layer_to_metrics_mapping[node_idx][bitwidth_idx] = max(metric_value, max_config_value + eps)

        # Finalize distance metric mapping
        self.finalize_distance_metric(layer_to_metrics_mapping)

        return layer_to_metrics_mapping

    def _get_mp_graph(self, graph: Graph, target_resource_utilization: ResourceUtilization) -> Tuple[Graph, bool]:
        """
        Get graph for mixed precision search. Virtual graph is built if bops is restricted and both activation and
        weights are configurable.

        Args:
            graph: input graph.
            target_resource_utilization: target resource utilization.

        Returns:
            Graph for mixed precision search (virtual or original), and a boolean flag whether a virtual graph has been
            constructed.
        """
        if (target_resource_utilization.bops_restricted() and
                graph.has_any_configurable_activation() and
                graph.has_any_configurable_weights()):
            mp_graph = substitute(copy.deepcopy(graph),
                                  self.fw_impl.get_substitutions_virtual_weights_activation_coupling())
            return mp_graph, True

        return graph, False

    def get_search_space(self) -> Dict[int, List[int]]:
        """
        The search space is a mapping from a node's index to a list of integers (possible bitwidths candidates indeces
        for the node).

        Returns:
            The entire search space of the graph.
        """

        indices_mapping = {}
        for idx, n in enumerate(self.mp_topo_configurable_nodes):
            # For each node, get all possible bitwidth indices for it
            # (which is a list from 0 to the length of the candidates mp_config list of the node).
            indices_mapping[idx] = list(range(len(n.candidates_quantization_cfg)))  # all search_methods space
        return indices_mapping

    def _compute_relative_ru_matrices(self) -> Dict[RUTarget, np.ndarray]:
        """
        Computes and builds a resource utilization matrix for all restricted targets, to be used for the
        mixed-precision search problem formalization.
        Utilization is computed relative to the minimal configuration, i.e. utilization for it will be 0.

        Returns:
            A dictionary containing resource utilization matrix of shape (num configurations, num memory elements)
            per ru target. Num memory elements depends on the target, e.g. num cuts or 1 for cumulative metrics.
        """
        rus_per_candidate = defaultdict(list)
        for c, c_n in enumerate(self.mp_topo_configurable_nodes):
            for candidate_idx in range(len(c_n.candidates_quantization_cfg)):
                if candidate_idx == self.min_ru_config[c]:
                    candidate_rus = self.min_ru
                else:
                    candidate_rus = self.compute_ru_for_candidate(c, candidate_idx)

                for target, ru in candidate_rus.items():
                    rus_per_candidate[target].append(ru)

        # Each target contains a matrix of num configurations X num elements
        relative_rus = {target: np.array(ru) - self.min_ru[target] for target, ru in rus_per_candidate.items()}
        return relative_rus

    def compute_ru_for_candidate(self, conf_node_idx: int, candidate_idx: int) -> Dict[RUTarget, np.ndarray]:
        """
        Computes a resource utilization vector after replacing the given node's configuration candidate in the minimal
        target configuration with the given candidate index.

        Args:
            conf_node_idx: The index of a node in a sorted configurable nodes list.
            candidate_idx: Quantization config candidate to be used for the node's resource utilization computation.

        Returns:
            Node's resource utilization vector.

        """
        cfg = self.replace_config_in_index(self.min_ru_config, conf_node_idx, candidate_idx)
        return self.ru_helper.compute_utilization(self.ru_targets, cfg)

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

    def compute_resource_utilization_for_config(self, config: List[int]) -> ResourceUtilization:
        """
        Computes the resource utilization values for a given mixed-precision configuration.

        Args:
            config: A mixed-precision configuration (list of candidates indices)

        Returns: A ResourceUtilization object with the model's resource utilization values when quantized 
        with the given config.

        """
        act_qcs, w_qcs = self.ru_helper.get_quantization_candidates(config)
        ru = self.ru_helper.ru_calculator.compute_resource_utilization(
            target_criterion=TargetInclusionCriterion.AnyQuantized, bitwidth_mode=BitwidthMode.QCustom, act_qcs=act_qcs,
            w_qcs=w_qcs, ru_targets=self.ru_targets, allow_unused_qcs=True)
        return ru

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
                Logger.critical(f"Virtual graph configuration error: Expected the predecessor of node '{weights_node.name}' to have multiple outputs when not composed with an activation node.")  # pragma: no cover

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
