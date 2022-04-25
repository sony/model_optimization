# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

from typing import Callable
from typing import Dict, List
import numpy as np

from model_compression_toolkit.common.constants import ACTIVATION, WEIGHTS
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.framework_info import FrameworkInfo


class MixedPrecisionSearchManager(object):
    """
    Class to wrap and manage the search process of a mixed-precision configuration.
    """

    def __init__(self,
                 graph: Graph,
                 qc: MixedPrecisionQuantizationConfig,
                 fw_info: FrameworkInfo,
                 get_sensitivity_evaluation: Callable,
                 compute_config_weights_kpi: Callable,
                 compute_config_activation_kpi: Callable,
                 kpi_weights_aggr_method: Callable,
                 kpi_activation_aggr_method: Callable):
        """

        Args:
            graph: Graph to search for its MP configuration.
            qc: Quantization configuration for how the graph should be quantized.
            fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
            get_sensitivity_evaluation: Framework specific function to retrieve a metric computation function.
        """

        self.graph = graph
        self.qc = qc
        self.fw_info = fw_info
        self.get_sensitivity_evaluation = get_sensitivity_evaluation
        self.metrics_weights = self.qc.distance_weighting_method
        self.layer_to_bitwidth_mapping = self.get_search_space()
        self.compute_metric_fn = self.get_sensitivity_metric()

        self.compute_config_kpi = {WEIGHTS: compute_config_weights_kpi, ACTIVATION: compute_config_activation_kpi}
        self.min_kpi_config = {WEIGHTS: self.get_min_weights_cfg(), ACTIVATION: self.get_min_activation_cfg()}
        min_kpi = self.compute_min_kpis()
        self.min_kpi = {WEIGHTS: min_kpi[0], ACTIVATION: min_kpi[1]}
        self.configurable_nodes_per_target = {WEIGHTS: self.graph.get_sorted_weights_configurable_nodes,
                                              ACTIVATION: self.graph.get_sorted_activation_configurable_nodes}

        self.kpi_aggr_methods = {WEIGHTS: kpi_weights_aggr_method,
                                ACTIVATION: kpi_activation_aggr_method}

    def get_search_space(self) -> Dict[int, List[int]]:
        """
        The search space is a mapping from a node's index to a list of integers (possible bitwidths candidates indeces
        for the node).

        Returns:
            The entire search space of the graph.
        """

        indices_mapping = {}
        nodes_to_configure = self.graph.get_configurable_sorted_nodes()
        for idx, n in enumerate(nodes_to_configure):
            # For each node, get all possible bitwidth indices for it
            # (which is a list from 0 to the length of the candidates qc list of the node).
            indices_mapping[idx] = list(range(len(n.candidates_quantization_cfg)))  # all search_methods space
        return indices_mapping

    def get_min_activation_cfg(self):
        """
        Builds a mixed-precision config with the bitwidth indexes for model with minimal activation KPI.

        Returns: A mp configuration (list of indices)

        """
        nodes_to_configure = self.graph.get_configurable_sorted_nodes()
        nodes_activation_bitwidth_candidates = [[c.activation_quantization_cfg.activation_n_bits for c in
                                                 n.candidates_quantization_cfg] for n in nodes_to_configure]
        return [np.argmin(n_candidates) for n_candidates in nodes_activation_bitwidth_candidates]

    def get_min_weights_cfg(self):
        """
        Builds a mixed-precision config with the bitwidth indexes for model with minimal weights KPI.

        Returns: A mp configuration (list of indices)

        """
        nodes_to_configure = self.graph.get_configurable_sorted_nodes()
        nodes_weights_bitwidth_candidates = [[c.weights_quantization_cfg.weights_n_bits for c in
                                              n.candidates_quantization_cfg] for n in nodes_to_configure]
        return [np.argmin(n_candidates) for n_candidates in nodes_weights_bitwidth_candidates]

    def get_sensitivity_metric(self) -> Callable:
        """

        Returns: Return a function (from the framework implementation) to compute a metric that
        indicates the similarity of the mixed-precision model (to the float model) for a given
        mixed-precision configuration.

        """
        # Get from the framework an evaluation function on how a MP configuration,
        # affects the expected loss.
        compute_metric_fn = self.get_sensitivity_evaluation(self.graph,
                                                            self.qc,
                                                            self.metrics_weights)
        return compute_metric_fn

    def compute_min_kpis(self):
        """
        Computes a KPIs vector with the values matching to the minimal mp configuration
            (i.e., each node is configured with the quantization candidate that would give the minimal size of the
            node's KPI).
        The method computes the minimal KPIs vector for minimal weights config and minimal activation config separately.

        Returns: A pair of KPIs vectors for weights and activation minimal configuration, respectively,

        """
        return self.compute_config_kpi[WEIGHTS](self.min_kpi_config[WEIGHTS], self.graph, self.fw_info), \
               self.compute_config_kpi[ACTIVATION](self.min_kpi_config[ACTIVATION], self.graph, self.fw_info)

    def compute_kpi_metrix(self, target):
        """
        Computes and builds a KPIs metrix, to be used for the mixed-precision search problem formalization.
        The matrix is constructed as follows (for a given target):
        - Each row represents the set of KPI values for a specific KPI measure (number of rows should be equal to the
            length of the output of the respective target compute_kpi function).
        - Each entry in a specific column represents the KPI value of a given configuration (single layer is configured
            with specific candidate, all other layer are at the minimal KPI configuration) for the KPI measure of the
            respective row.

        Args:
            target: The target for which the KPI is calculated (should be one of 'weights' or 'activation').

        Returns: A KPI matrix.

        """
        assert target == WEIGHTS or target == ACTIVATION, \
            f"{target} is not a valid KPI target, valid KPI targets are {[WEIGHTS, ACTIVATION]}"

        configurable_sorted_nodes = self.graph.get_configurable_sorted_nodes()

        kpi_matrix = []
        for c, c_n in enumerate(configurable_sorted_nodes):
            for candidate_idx in range(len(c_n.candidates_quantization_cfg)):
                candidate_kpis = self.compute_candidate_relative_kpis(c, candidate_idx, target)
                kpi_matrix.append(np.asarray(candidate_kpis))

        # We need to transpose the calculated kpi matrix to allow later multiplication with
        # the indicators' diagonal matrix
        return np.transpose(np.array(kpi_matrix))

    def compute_candidate_relative_kpis(self, conf_node_idx, candidate_idx, target):
        """
        Computes a KPIs vector for a given candidates of a given configurable node, i.e., the matching KPI vector
        which is obtained by computing the given target's KPI function on a minimal configuration in which the given
        layer's candidates is changed to the new given one.
        The result is normalized by subtracting the target's minimal KPIs vector.

        Args:
            conf_node_idx: The index of a node in a sorted configurable nodes list.
            candidate_idx: The index of a node's quantization configuration candidate.
            target: The target for which the KPI is calculated (should be one of 'weights' or 'activation').

        Returns: Normalized node's KPIs vector

        """
        return self.compute_node_kpi_for_candidate(conf_node_idx, candidate_idx, target) - self.get_min_target_kpi(target)

    def get_min_target_kpi(self, target):
        """
        Returns the minimal KPIs vector (pre-calculated on initialization) of a specific target.

        Args:
            target: The target for which the KPI is calculated (should be one of 'weights' or 'activation').

        Returns: Minimal KPIs vector.

        """
        return self.min_kpi[target]

    def compute_node_kpi_for_candidate(self, conf_node_idx, candidate_idx, target):
        """
        Computes a KPIs vector after replacing the given node's configuration candidate in the minimal
        target configuration with the given candidate index.

        Args:
            conf_node_idx: The index of a node in a sorted configurable nodes list.
            candidate_idx: Quantization config candidate to be used for the node's KPI computation.
            target: The target for which the KPI is calculated (should be one of 'weights' or 'activation').

        Returns: Node's KPIs vector.

        """
        return self.compute_config_kpi[target](
            self.replace_config_in_index(
                self.min_kpi_config[target],
                conf_node_idx,
                candidate_idx),
            self.graph,
            self.fw_info)

    @staticmethod
    def replace_config_in_index(mp_cfg, idx, value):
        """
        Replacing the quantization configuration candidate in a minimal mixed-precision configuration at the given
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
