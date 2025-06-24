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
import os

import itertools

import copy
from collections import defaultdict

from tqdm import tqdm

from typing import Dict, List, Tuple, Optional

import numpy as np

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode, \
    VirtualSplitWeightsNode, VirtualSplitActivationNode, VirtualNode
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget, ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    TargetInclusionCriterion, BitwidthMode
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_ru_helper import \
    MixedPrecisionRUHelper
from model_compression_toolkit.core.common.mixed_precision.search_methods.linear_programming import \
    MixedPrecisionIntegerLPSolver
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.sensitivity_evaluation import SensitivityEvaluation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig, MpMetricNormalization


class MixedPrecisionSearchManager:
    """
    Class to wrap and manage the search process of a mixed-precision configuration.
    """

    def __init__(self,
                 graph: Graph,
                 fw_impl: FrameworkImplementation,
                 sensitivity_evaluator: SensitivityEvaluation,
                 target_resource_utilization: ResourceUtilization,
                 mp_config: MixedPrecisionQuantizationConfig):
        """

        Args:
            graph: Graph to search for its MP configuration.
            fw_impl: FrameworkImplementation object with specific framework methods implementation.
            sensitivity_evaluator: A SensitivityEvaluation which provides a function that evaluates the sensitivity of
                a bit-width configuration for the MP model.
            target_resource_utilization: Target Resource Utilization to bound our feasible solution space s.t the configuration does not violate it.
        """

        self.fw_impl = fw_impl

        self.original_graph = graph
        # graph for mp search
        self.mp_graph, self.using_virtual_graph = self._get_mp_graph(graph, target_resource_utilization)
        del graph  # so that it's not used by mistake

        self.sensitivity_evaluator = sensitivity_evaluator
        self.target_resource_utilization = target_resource_utilization
        self.mp_config = mp_config

        self.mp_topo_configurable_nodes = self.mp_graph.get_configurable_sorted_nodes()

        self.ru_targets = target_resource_utilization.get_restricted_targets()
        self.orig_graph_ru_helper = MixedPrecisionRUHelper(self.original_graph, fw_impl)

        self.min_ru_config: Dict[BaseNode, int] = self.mp_graph.get_min_candidates_config()

        self.config_reconstructor = None
        orig_min_config = self.min_ru_config
        if self.using_virtual_graph:
            self.config_reconstructor = ConfigReconstructionHelper(self.original_graph)
            orig_min_config = self.config_reconstructor.reconstruct_full_configuration(self.min_ru_config)
        self.min_ru = self.orig_graph_ru_helper.compute_utilization(self.ru_targets, orig_min_config)

    def search(self) -> Dict[BaseNode, int]:
        """
        Run mixed precision search.

        Returns:
            Mapping from nodes to indices of the selected bit-widths candidate.
        """
        mp_config = self._prepare_and_run_solver()

        if self.using_virtual_graph:
            mp_config = self.config_reconstructor.reconstruct_full_configuration(mp_config)

        return mp_config

    def _prepare_and_run_solver(self) -> Dict[BaseNode, int]:
        """
        Prepare sensitivity and ru data for LP solver and run the solver.

        Returns:
            Mapping from nodes to indices of the selected bit-widths candidate.
        """
        candidates_ru = self._compute_relative_ru_matrices()
        rel_target_ru = self._get_relative_ru_constraint_per_mem_element()
        layers_candidates_sensitivity: Dict[BaseNode, List[float]] = self._build_sensitivity_mapping()
        solver = MixedPrecisionIntegerLPSolver(layers_candidates_sensitivity, candidates_ru, rel_target_ru)
        mp_config = solver.run()
        return mp_config

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
            ru_target: (ru - self.min_ru[ru_target]) for ru_target, ru in target_ru.items()
        }
        unsatisfiable_targets = {
            ru_target.value: target_ru[ru_target] for ru_target, ru in rel_target_ru.items() if any(ru < 0)
        }
        if unsatisfiable_targets:
            raise ValueError(f"The model cannot be quantized to meet the specified resource utilization for the "
                             f"following targets: {unsatisfiable_targets}")
        return rel_target_ru

    def _build_sensitivity_mapping(self) -> Dict[BaseNode, List[float]]:
        """
        This function measures the sensitivity of a change in a bitwidth of a layer on the entire model.

        Returns:
            Mapping from nodes to their bitwidth candidates sensitivity.
        """
        Logger.info('Starting to evaluate metrics')
        norm_method = self.mp_config.metric_normalization
        eps = self.mp_config.metric_epsilon

        verbose = 'VERBOSE_MP_METRIC' in os.environ

        def normalize(node_candidates_metrics, max_ind):
            if norm_method == MpMetricNormalization.NONE:
                return node_candidates_metrics
            if norm_method == MpMetricNormalization.MAXBIT:
                ref_ind = max_ind
            elif norm_method == MpMetricNormalization.MINBIT:
                ref_ind = node.find_min_candidate_index()
            else:  # pragma: no cover
                raise ValueError(f'Unexpected MpMetricNormalization mode {norm_method}')
            normalized_metrics = node_candidates_metrics / node_candidates_metrics[ref_ind]
            return normalized_metrics

        def ensure_maxbit_minimal_metric(node_candidates_metrics, max_ind):
            if eps is None:
                return node_candidates_metrics
            # We want maxbit configuration to have the minimal distance metric (so that optimization objective
            # doesn't prefer lower bits). If we got a smaller metric for non-maxbit, we update it to metric(maxbit)+eps.
            max_val = node_candidates_metrics[max_ind]
            metrics = np.maximum(node_candidates_metrics, max_val + eps)
            metrics[max_ind] = max_val
            return metrics

        layer_to_metrics_mapping = {}
        debug_mapping = {}
        for node_idx, node in tqdm(enumerate(self.mp_topo_configurable_nodes)):
            raw_candidates_sensitivity = np.empty(len(node.candidates_quantization_cfg))
            for bitwidth_idx, _ in enumerate(node.candidates_quantization_cfg):
                if self.using_virtual_graph:
                    a_cfg, w_cfg = self.config_reconstructor.reconstruct_separate_aw_configs({node: bitwidth_idx})
                else:
                    a_cfg = {node: bitwidth_idx} if node.has_configurable_activation() else {}
                    w_cfg = {node: bitwidth_idx} if node.has_any_configurable_weight() else {}
                raw_candidates_sensitivity[bitwidth_idx] = self.sensitivity_evaluator.compute_metric(
                    mp_a_cfg={n.name: ind for n, ind in a_cfg.items()},
                    mp_w_cfg={n.name: ind for n, ind in w_cfg.items()}
                )
            max_ind = node.find_max_candidate_index()
            normalized_sensitivity = normalize(raw_candidates_sensitivity, max_ind)
            candidates_sensitivity = ensure_maxbit_minimal_metric(normalized_sensitivity, max_ind)
            layer_to_metrics_mapping[node] = candidates_sensitivity

            if verbose:    # pragma: no cover
                debug_mapping[node] = {'': candidates_sensitivity}
                if np.any(raw_candidates_sensitivity != candidates_sensitivity):
                    debug_mapping[node]['normalized'] = normalized_sensitivity
                    debug_mapping[node]['raw       '] = raw_candidates_sensitivity

        if verbose:    # pragma: no cover
            np.set_printoptions(precision=8, floatmode='maxprec')
            name_len = max(len(n.name) for n in debug_mapping)
            s = '\nMETRIC BEGIN'
            for n, d in debug_mapping.items():
                s += (f'\n{n.name:{name_len}}' + f'\n{" ":{name_len-10}}'.join([f'{k} {v}' for k, v in d.items()]))
            s += '\nMETRIC END'
            Logger.info(s)
        # Finalize distance metric mapping
        self._finalize_distance_metric(layer_to_metrics_mapping)

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
        for node in self.mp_topo_configurable_nodes:
            for candidate_idx, _ in enumerate(node.candidates_quantization_cfg):
                if candidate_idx == self.min_ru_config[node]:
                    candidate_rus = self.min_ru
                else:
                    cfg = self.min_ru_config.copy()
                    cfg[node] = candidate_idx
                    if self.using_virtual_graph:
                        cfg = self.config_reconstructor.reconstruct_full_configuration(cfg)
                    candidate_rus = self.orig_graph_ru_helper.compute_utilization(self.ru_targets, cfg)

                for target, ru in candidate_rus.items():
                    rus_per_candidate[target].append(ru)

        # Each target contains a matrix of num configurations X num elements
        relative_rus = {target: (np.array(ru) - self.min_ru[target]) for target, ru in rus_per_candidate.items()}
        return relative_rus

    @staticmethod
    def copy_config_with_replacement(mp_cfg: Dict[BaseNode, int], node: BaseNode, candidate_idx: int) -> Dict[BaseNode, int]:
        """
        Create a copy of the given mixed-precision configuration and update the candidate index for a specific node.

        Args:
            mp_cfg: Mixed-precision configuration.
            node: Node to update the config for.
            candidate_idx: A new candidate index to configure.

        Returns:
            A new mixed-precision configuration.

        """
        updated_cfg = mp_cfg.copy()
        updated_cfg[node] = candidate_idx
        return updated_cfg

    def compute_resource_utilization_for_config(self, config: Dict[BaseNode, int]) -> ResourceUtilization:
        """
        Computes the resource utilization values for a given mixed-precision configuration.

        Args:
            config: A mixed-precision configuration (list of candidates indices)

        Returns: A ResourceUtilization object with the model's resource utilization values when quantized 
        with the given config.

        """
        act_qcs, w_qcs = self.orig_graph_ru_helper.get_quantization_candidates(config)
        ru = self.orig_graph_ru_helper.ru_calculator.compute_resource_utilization(
            target_criterion=TargetInclusionCriterion.AnyQuantizedNonFused,
            bitwidth_mode=BitwidthMode.QCustom,
            act_qcs=act_qcs,
            w_qcs=w_qcs,
            ru_targets=self.ru_targets,
            allow_unused_qcs=True)
        return ru

    def _finalize_distance_metric(self, layer_to_metrics_mapping: Dict[BaseNode, List[float]]):
        """
        Finalizing the distance metric building.
        The method checks to see if the maximal distance value is larger than a given threshold, and if so,
        it scales all metric values to prevent possible numerical issues.
        Modification to the dictionary is done inplace.

        Args:
            layer_to_metrics_mapping: A mapping between a node to a list of distance values per bitwidth candidate.

        """
        # normalize metric for numerical stability
        max_dist = max(itertools.chain.from_iterable(layer_to_metrics_mapping.values()))

        if max_dist >= self.mp_config.metric_normalization_threshold:
            Logger.warning(f"The mixed precision distance metric values indicate a large error in the quantized model."
                           f"this can cause numerical issues."
                           f"The program will proceed with mixed precision search after scaling the metric values,"
                           f"which can lead to unstable results.")
            for layer, dists in layer_to_metrics_mapping.items():
                for i, _ in enumerate(dists):
                    layer_to_metrics_mapping[layer][i] /= max_dist


class ConfigReconstructionHelper:
    def __init__(self, original_graph):
        # mapping in order to return the actual node objects from the original graph
        self.orig_nodes = {n.name: n for n in original_graph.nodes}

    def reconstruct_full_configuration(self,
                                       virtual_cfg: Dict[BaseNode, int],
                                       include_non_configurable: bool = False) -> Dict[BaseNode, int]:
        """
        Convert a configuration of a virtual graph into the corresponding configuration of the original graph.
        Note that a configurable VirtualActivationWeightsNode might comprise one configurable and one non-configurable
        original nodes.

        Args:
            virtual_cfg: a mapping from nodes in the virtual graph to selected candidate index. Should contain all
                configurable nodes of the virtual graph, and only configurable nodes.
            include_non_configurable: whether to return configs for non-configurable original nodes.

        Returns:
            A mapping from configurable nodes in the original graph to their candidate indices.
        """
        # Original candidate of a node that has been split might be determined by two different virtual nodes, one
        # determines activation and one - weights. First, for each virtual node we collect the original
        # activation / weights nodes, with all original candidates that match the virtual candidate
        # activation / weights config. If both activation and weights of the original node are determined by virtual
        # candidates, we look for a common candidate.
        orig_nodes_a_candidates = {}
        orig_nodes_w_candidates = {}
        for virtual_node, virtual_qc_ind in virtual_cfg.items():
            assert virtual_node.has_configurable_activation() or virtual_node.has_any_configurable_weight()
            orig_a_node, orig_a_candidates = self._retrieve_matching_orig_a_candidates(virtual_node, virtual_qc_ind)
            if orig_a_node and (include_non_configurable or orig_a_node.has_configurable_activation()):
                assert orig_a_node not in orig_nodes_a_candidates
                orig_nodes_a_candidates[orig_a_node] = orig_a_candidates
            orig_w_node, orig_w_candidates = self._retrieve_matching_orig_w_candidates(virtual_node, virtual_qc_ind)
            if orig_w_node and (include_non_configurable or orig_w_node.has_any_configurable_weight()):
                assert orig_w_node not in orig_nodes_w_candidates
                orig_nodes_w_candidates[orig_w_node] = orig_w_candidates

        orig_cfg = {}
        common_orig_nodes = set(orig_nodes_a_candidates.keys()).intersection(set(orig_nodes_w_candidates))
        for orig_node in common_orig_nodes:
            a_candidates = orig_nodes_a_candidates[orig_node]
            w_candidates = orig_nodes_w_candidates[orig_node]
            # find the common candidate
            common_candidates = set(a_candidates).intersection(set(w_candidates))
            if len(common_candidates) != 1:    # pragma: no cover
                raise ValueError(f'Expected to find exactly one candidate with the required activation and weights '
                                 f'quantization configuration for node {orig_node}. Found {len(common_candidates)}')
            # in theory it's possible that original non-configurable node gets split and each part is combined
            # with a configurable part of another node and we end up here
            if orig_node.has_configurable_activation() or orig_node.has_any_configurable_weight():
                orig_cfg[orig_node] = common_candidates.pop()
            del orig_nodes_a_candidates[orig_node]
            del orig_nodes_w_candidates[orig_node]

        # remaining a nodes
        for orig_node, a_candidates in orig_nodes_a_candidates.items():
            assert not orig_node.has_any_configurable_weight()  # if it had we should have caught it above
            assert len(a_candidates) == 1
            assert orig_node not in orig_cfg
            if include_non_configurable or orig_node.has_configurable_activation():
                orig_cfg[orig_node] = a_candidates[0]

        # remaining w nodes
        for orig_node, w_candidates in orig_nodes_w_candidates.items():
            assert not orig_node.has_configurable_activation()  # if it had we should have caught it above
            assert len(w_candidates) == 1
            assert orig_node not in orig_cfg
            if include_non_configurable or orig_node.has_any_configurable_weight():
                orig_cfg[orig_node] = w_candidates[0]

        return orig_cfg

    def reconstruct_separate_aw_configs(self,
                                        virtual_cfg: Dict[BaseNode, int],
                                        include_non_configurable: bool = False) \
            -> Tuple[Dict[BaseNode, int], Dict[BaseNode, int]]:
        """
        Retrieves original activation and weights nodes and corresponding candidates for a given configuration of the
        virtual graph. Only returns configuration specified by the virtual config, per configurable target (activation
        or weights). For example, if 'virtual_cfg' contains a single VirtualActivationWeightsNode, the returned
        configuration will contain only activation config for the original activation node, and only weights config
        for the original weights node).
        In practice, we return candidate index in both cases, instead of actual activation or weights config, since
        sensitivity evaluator heavily depends on it, so we must ignore activation config in weights candidate and vice
        versa. This is bad!!! TODO

        Args:
            virtual_cfg: a mapping from nodes in the virtual graph to selected candidate index.
            include_non_configurable: whether to return configs for non-configurable target (i.e. activation config
              for non-configurable activation, and weights config for non-configurable weight).

        Returns:
            Configuration for original activation nodes and a separate configuration for original weights nodes.
        """
        a_cfg = {}
        w_cfg = {}
        for virtual_node, virtual_qc_ind in virtual_cfg.items():
            orig_a_node, orig_a_candidates = self._retrieve_matching_orig_a_candidates(virtual_node, virtual_qc_ind)
            if orig_a_node and (include_non_configurable or orig_a_node.has_configurable_activation()):
                # we may have retrieved multiple candidates with different weights configs and identical activation
                # configs, so we just take the first
                a_cfg[orig_a_node] = orig_a_candidates[0]

            orig_w_node, orig_w_candidates = self._retrieve_matching_orig_w_candidates(virtual_node, virtual_qc_ind)
            if orig_w_node and (include_non_configurable or orig_w_node.has_any_configurable_weight()):
                # we may have retrieved multiple candidates with different activation configs and identical weights
                # configs, so we just take the first
                w_cfg[orig_w_node] = orig_w_candidates[0]

        return a_cfg, w_cfg

    def _retrieve_matching_orig_a_candidates(self,
                                             virtual_node: BaseNode,
                                             virtual_qc_ind: int) -> Tuple[Optional[BaseNode], Optional[List[int]]]:
        """
        Retrieve the original activation node and all its candidates matching activation quantization config of the
        given virtual candidate (candidate of a node in the virtual graph).
        Note that we do simple matching, without any filtering, so disabled activation quantization will be also matched.

        Args:
            virtual_node: node in the virtual graph (can be virtual or regular).
            virtual_qc_ind: candidate index of the virtual node.

        Returns:
            The original activation node (actual object from the original graph) and a list of its matching candidates.
        """
        if not isinstance(virtual_node, VirtualNode):
            return self.orig_nodes[virtual_node.name], [virtual_qc_ind]
        if isinstance(virtual_node, VirtualSplitWeightsNode):
            return None, None
        if isinstance(virtual_node, VirtualActivationWeightsNode):
            orig_a_node = virtual_node.original_activation_node
            if isinstance(orig_a_node, VirtualSplitActivationNode):
                orig_a_node = orig_a_node.origin_node
        else:
            assert isinstance(virtual_node, VirtualSplitActivationNode)
            orig_a_node = virtual_node.origin_node

        virtual_qc = virtual_node.candidates_quantization_cfg[virtual_qc_ind]
        matching_orig_a_cfgs = [i for i, orig_qc in enumerate(orig_a_node.candidates_quantization_cfg)
                                if orig_qc.activation_quantization_cfg == virtual_qc.activation_quantization_cfg]
        if not matching_orig_a_cfgs:    # pragma: no cover
            raise ValueError(f'Could not find matching activation quantization config in the original node '
                             f'{orig_a_node} for candidate {virtual_qc_ind} of the virtual node {virtual_node}')
        return self.orig_nodes[orig_a_node.name], matching_orig_a_cfgs

    def _retrieve_matching_orig_w_candidates(self,
                                             virtual_node: BaseNode,
                                             virtual_qc_ind: int) -> Tuple[Optional[BaseNode], Optional[List[int]]]:
        """
        Retrieve the original weights node and all its candidates matching weights quantization config of the
        given virtual candidate (candidate of a node in the virtual graph).

        Args:
            virtual_node: node in the virtual graph (can be virtual or regular).
            virtual_qc_ind: candidate index of the virtual node.

        Returns:
            The original weights node (actual object from the original graph) and a list of all its matching candidates.
        """
        if not isinstance(virtual_node, VirtualNode):
            if virtual_node.weights:
                return self.orig_nodes[virtual_node.name], [virtual_qc_ind]
            return None, None
        if isinstance(virtual_node, VirtualSplitActivationNode):
            return None, None

        if isinstance(virtual_node, VirtualActivationWeightsNode):
            assert isinstance(virtual_node.original_weights_node, VirtualSplitWeightsNode)
            orig_w_node = virtual_node.original_weights_node.origin_node
        else:
            assert isinstance(virtual_node, VirtualSplitWeightsNode)
            orig_w_node = virtual_node.origin_node

        virtual_qc = virtual_node.candidates_quantization_cfg[virtual_qc_ind]

        # Matching candidate is a candidate with matching configs for configurable weights. We cannot compare the entire
        # weights config since the virtual node may contain additional non-configurable weights from the activation node
        orig_configurable_attrs = [attr for attr in orig_w_node.weights if virtual_node.is_configurable_weight(attr)]
        assert all(virtual_node.is_configurable_weight(attr) for attr in orig_configurable_attrs)

        def get_configurable_attrs_cfgs(qc):
            return {attr: qc.weights_quantization_cfg.get_attr_config(attr) for attr in orig_configurable_attrs}
        virtual_cfg = get_configurable_attrs_cfgs(virtual_qc)
        matching_orig_w_cfgs = [i for i, orig_qc in enumerate(orig_w_node.candidates_quantization_cfg)
                                if get_configurable_attrs_cfgs(orig_qc) == virtual_cfg]
        if not matching_orig_w_cfgs:    # pragma: no cover
            raise ValueError(f'Could not find matching weights quantization config in the original node '
                             f'{orig_w_node} for candidate {virtual_qc_ind} of the virtual node {virtual_node}')
        return self.orig_nodes[orig_w_node.name], matching_orig_w_cfgs
