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

from typing import Callable, Tuple
from typing import Dict, List
import numpy as np

from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.mixed_precision.kpi import KPITarget
from model_compression_toolkit.core.common.mixed_precision.kpi_aggregation_methods import MpKpiAggregation
from model_compression_toolkit.core.common.mixed_precision.kpi_methods import MpKpiMetric
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import SensitivityEvaluation


class MixedPrecisionSearchManager:
    """
    Class to wrap and manage the search process of a mixed-precision configuration.
    """

    def __init__(self,
                 graph: Graph,
                 fw_info: FrameworkInfo,
                 sensitivity_evaluator: SensitivityEvaluation,
                 kpi_functions: Dict[KPITarget, Tuple[MpKpiMetric, MpKpiAggregation]]):
        """

        Args:
            graph: Graph to search for its MP configuration.
            fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
            sensitivity_evaluator: A SensitivityEvaluation which provides a function that evaluates the sensitivity of
                a bit-width configuration for the MP model.
            kpi_functions: A dictionary with pairs of (MpKpiMethod, MpKpiAggregationMethod) mapping a KPITarget to
                a couple of kpi metric function and kpi aggregation function.
        """

        self.graph = graph
        self.fw_info = fw_info
        self.sensitivity_evaluator = sensitivity_evaluator
        self.layer_to_bitwidth_mapping = self.get_search_space()
        self.compute_metric_fn = self.get_sensitivity_metric()

        self.compute_kpi_functions = kpi_functions

        self.min_kpi_config = self.graph.get_min_candidates_config()
        self.max_kpi_config = self.graph.get_max_candidates_config()

        self.min_kpi = self.compute_min_kpis()

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

    def compute_min_kpis(self) -> Dict[KPITarget, np.ndarray]:
        """
        Computes a KPIs vector with the values matching to the minimal mp configuration
        (i.e., each node is configured with the quantization candidate that would give the minimal size of the
        node's KPI).
        The method computes the minimal KPIs vector for each kpi target.

        Returns: A dictionary mapping each kpi target to its respective minimal KPIs values.

        """
        min_kpis = {}
        for kpi_target, kpi_fns in self.compute_kpi_functions.items():
            # kpi_fns is a pair of kpi computation method and kpi aggregation method (in this method we only need
            # the first one)
            min_kpis[kpi_target] = kpi_fns[0](self.min_kpi_config, self.graph, self.fw_info)

        return min_kpis

    def compute_kpi_matrix(self, target: KPITarget) -> np.ndarray:
        """
        Computes and builds a KPIs matrix, to be used for the mixed-precision search problem formalization.
        The matrix is constructed as follows (for a given target):
        - Each row represents the set of KPI values for a specific KPI measure (number of rows should be equal to the
            length of the output of the respective target compute_kpi function).
        - Each entry in a specific column represents the KPI value of a given configuration (single layer is configured
            with specific candidate, all other layer are at the minimal KPI configuration) for the KPI measure of the
            respective row.

        Args:
            target: The target for which the KPI is calculated (a KPITarget value).

        Returns: A KPI matrix.

        """
        assert isinstance(target, KPITarget), f"{target} is not a valid KPI target"

        configurable_sorted_nodes = self.graph.get_configurable_sorted_nodes()

        kpi_matrix = []
        for c, c_n in enumerate(configurable_sorted_nodes):
            for candidate_idx in range(len(c_n.candidates_quantization_cfg)):
                candidate_kpis = self.compute_candidate_relative_kpis(c, candidate_idx, target)
                kpi_matrix.append(np.asarray(candidate_kpis))

        # We need to transpose the calculated kpi matrix to allow later multiplication with
        # the indicators' diagonal matrix.
        # We only move the first axis (num of configurations) to be last,
        # the remaining axes include the metric specific nodes (rows dimension of the new tensor)
        # and the kpi metric values (if they are non-scalars)
        np_kpi_matrix = np.array(kpi_matrix)
        return np.moveaxis(np_kpi_matrix, source=0, destination=len(np_kpi_matrix.shape) - 1)

    def compute_candidate_relative_kpis(self,
                                        conf_node_idx: int,
                                        candidate_idx: int,
                                        target: KPITarget) -> np.ndarray:
        """
        Computes a KPIs vector for a given candidates of a given configurable node, i.e., the matching KPI vector
        which is obtained by computing the given target's KPI function on a minimal configuration in which the given
        layer's candidates is changed to the new given one.
        The result is normalized by subtracting the target's minimal KPIs vector.

        Args:
            conf_node_idx: The index of a node in a sorted configurable nodes list.
            candidate_idx: The index of a node's quantization configuration candidate.
            target: The target for which the KPI is calculated (a KPITarget value).

        Returns: Normalized node's KPIs vector

        """
        return self.compute_node_kpi_for_candidate(conf_node_idx, candidate_idx, target) - \
               self.get_min_target_kpi(target)

    def get_min_target_kpi(self, target: KPITarget) -> np.ndarray:
        """
        Returns the minimal KPIs vector (pre-calculated on initialization) of a specific target.

        Args:
            target: The target for which the KPI is calculated (a KPITarget value).

        Returns: Minimal KPIs vector.

        """
        return self.min_kpi[target]

    def compute_node_kpi_for_candidate(self, conf_node_idx: int, candidate_idx: int, target: KPITarget) -> np.ndarray:
        """
        Computes a KPIs vector after replacing the given node's configuration candidate in the minimal
        target configuration with the given candidate index.

        Args:
            conf_node_idx: The index of a node in a sorted configurable nodes list.
            candidate_idx: Quantization config candidate to be used for the node's KPI computation.
            target: The target for which the KPI is calculated (a KPITarget value).

        Returns: Node's KPIs vector.

        """
        return self.compute_kpi_functions[target][0](
            self.replace_config_in_index(
                self.min_kpi_config,
                conf_node_idx,
                candidate_idx),
            self.graph,
            self.fw_info)

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
