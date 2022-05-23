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

import numpy as np
from pulp import *
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable

from model_compression_toolkit.common import Logger
from model_compression_toolkit.common.mixed_precision.kpi import KPI, KPITarget
from model_compression_toolkit.common.mixed_precision.mixed_precision_search_manager import MixedPrecisionSearchManager


def mp_integer_programming_search(search_manager: MixedPrecisionSearchManager,
                                  target_kpi: KPI = None) -> List[int]:
    """
    Searching and returning a mixed-precision configuration using an ILP optimization solution.
    It first builds a mapping from each layer's index (in the model) to a dictionary that maps the
    bitwidth index to the observed sensitivity of the model when using that bitwidth for that layer.
    Then, it creates a mapping from each node's index (in the graph) to a dictionary
    that maps the bitwidth index to the contribution of configuring this node with this
    bitwidth to the minimal possible KPI of the model.
    Then, and using these mappings, it builds an LP problem and finds an optimal solution.
    If a solution could not be found, exception is thrown.

    Args:
        search_manager: MixedPrecisionSearchManager object to be used for problem formalization.
        target_kpi: KPI to constrain our LP problem with some resources limitations (like model' weights memory
        consumption).

    Returns:
        The mixed-precision configuration (list of indices. Each indicates the bitwidth index of a node).

    """

    # Build a mapping from each layer's index (in the model) to a dictionary that maps the
    # bitwidth index to the observed sensitivity of the model when using that bitwidth for that layer.
    layer_to_metrics_mapping = _build_layer_to_metrics_mapping(search_manager.layer_to_bitwidth_mapping,
                                                               search_manager.compute_metric_fn,
                                                               search_manager.max_kpi_config)

    # Init variables to find their values when solving the lp problem.
    layer_to_indicator_vars_mapping, layer_to_objective_vars_mapping = _init_problem_vars(layer_to_metrics_mapping)

    # Add all equations and inequalities that define the problem.
    lp_problem = _formalize_problem(layer_to_indicator_vars_mapping,
                                    layer_to_metrics_mapping,
                                    layer_to_objective_vars_mapping,
                                    target_kpi,
                                    search_manager)

    lp_problem.solve()  # Try to solve the problem.
    assert lp_problem.status == LpStatusOptimal, Logger.critical(
        "No solution was found during solving the LP problem")
    Logger.info(LpStatus[lp_problem.status])

    # Take the bitwidth index only if its corresponding indicator is one.
    config = np.asarray(
        [[nbits for nbits, indicator in nbits_to_indicator.items() if indicator.varValue == 1.0] for
         nbits_to_indicator
         in layer_to_indicator_vars_mapping.values()]
    ).flatten()

    return config


def _init_problem_vars(layer_to_metrics_mapping: Dict[int, Dict[int, float]]) -> Tuple[
    Dict[int, Dict[int, LpVariable]], Dict[int, LpVariable]]:
    """
    Initialize the LP problem variables: Variable for each layer as to the index of the bitwidth it should use,
    and a variable for each indicator for whether we use the former variable or not.

    Args:
        layer_to_metrics_mapping: Mapping from each layer's index (in the model) to a dictionary that maps the
        bitwidth index to the observed sensitivity of the model.

    Returns:
        A tuple of two dictionaries: One from a layer to the variable for the bitwidth problem,
        and the second for indicators for each variable.
    """

    layer_to_indicator_vars_mapping = dict()
    layer_to_objective_vars_mapping = dict()

    for layer, nbits_to_metric in layer_to_metrics_mapping.items():
        layer_to_indicator_vars_mapping[layer] = dict()

        for nbits in nbits_to_metric.keys():
            layer_to_indicator_vars_mapping[layer][nbits] = LpVariable(f"layer_{layer}_{nbits}",
                                                                       lowBound=0,
                                                                       upBound=1,
                                                                       cat=LpInteger)

        layer_to_objective_vars_mapping[layer] = LpVariable(f"s_{layer}", 0)

    return layer_to_indicator_vars_mapping, layer_to_objective_vars_mapping


def _formalize_problem(layer_to_indicator_vars_mapping: Dict[int, Dict[int, LpVariable]],
                       layer_to_metrics_mapping: Dict[int, Dict[int, float]],
                       layer_to_objective_vars_mapping: Dict[int, LpVariable],
                       target_kpi: KPI,
                       search_manager: MixedPrecisionSearchManager) -> LpProblem:
    """
    Formalize the LP problem by defining all inequalities that define the solution space.

    Args:
        layer_to_indicator_vars_mapping: Dictionary that maps each node's index to a dictionary of bitwidth to
        indicator variable.
        layer_to_metrics_mapping: Dictionary that maps each node's index to a dictionary of bitwidth to sensitivity
        evaluation.
        layer_to_objective_vars_mapping: Dictionary that maps each node's index to a bitwidth variable we find its
        value.
        target_kpi: KPI to reduce our feasible solution space.
        search_manager: MixedPrecisionSearchManager object to be used for kpi constraints formalization.

    Returns:
        The formalized LP problem.
    """

    lp_problem = LpProblem()  # minimization problem by default
    lp_problem += lpSum([layer_to_objective_vars_mapping[layer] for layer in
                         layer_to_metrics_mapping.keys()])  # Objective (minimize acc loss)

    for layer in layer_to_metrics_mapping.keys():
        # Use every bitwidth for every layer with its indicator.
        lp_problem += lpSum([indicator * layer_to_metrics_mapping[layer][nbits]
                             for nbits, indicator in layer_to_indicator_vars_mapping[layer].items()]) == \
                      layer_to_objective_vars_mapping[layer]

        # Constraint of only one indicator==1
        lp_problem += lpSum(
            [v for v in layer_to_indicator_vars_mapping[layer].values()]) == 1

    # Bound the feasible solution space with the desired KPI.
    # Creates separate constraints for weights KPI and activation KPI.
    if target_kpi is not None:
        indicators = []
        for layer in layer_to_metrics_mapping.keys():
            for _, indicator in layer_to_indicator_vars_mapping[layer].items():
                indicators.append(indicator)

        indicators_arr = np.array(indicators)
        indicators_matrix = np.diag(indicators_arr)

        if not np.isinf(target_kpi.weights_memory):
            _add_set_of_kpi_constraints(search_manager=search_manager,
                                        target=KPITarget.WEIGHTS,
                                        target_memory=target_kpi.weights_memory,
                                        indicators_matrix=indicators_matrix,
                                        lp_problem=lp_problem)

        if not np.isinf(target_kpi.activation_memory):
            _add_set_of_kpi_constraints(search_manager=search_manager,
                                        target=KPITarget.ACTIVATION,
                                        target_memory=target_kpi.activation_memory,
                                        indicators_matrix=indicators_matrix,
                                        lp_problem=lp_problem)

        if not np.isinf(target_kpi.total_memory):
            _add_set_of_kpi_constraints(search_manager=search_manager,
                                        target=KPITarget.TOTAL,
                                        target_memory=target_kpi.total_memory,
                                        indicators_matrix=indicators_matrix,
                                        lp_problem=lp_problem)

    else:
        raise Exception("Can't run mixed-precision search with given target_kpi=None."
                        "Please provide a valid target_kpi.")
    return lp_problem


def _add_set_of_kpi_constraints(search_manager: MixedPrecisionSearchManager,
                                target: KPITarget,
                                target_memory: float,
                                indicators_matrix: np.ndarray,
                                lp_problem: LpProblem):

    kpi_matrix = search_manager.compute_kpi_matrix(target)
    indicated_kpi_matrix = np.matmul(kpi_matrix, indicators_matrix)
    # Need to re-organize the tensor such that the configurations' axis will be second,
    # and all metric values' axis will come afterword
    indicated_kpi_matrix = np.moveaxis(indicated_kpi_matrix, source=len(indicated_kpi_matrix.shape) - 1, destination=1)

    # In order to get the result KPI according to a chosen set of indicators, we sum each row in the result matrix.
    # Each row represents the KPI values for a specific KPI metric, such that only elements corresponding
    # to a configuration which implied by the set of indicators will have some positive value different than 0
    # (and will contribute to the total KPI).
    kpi_sum_vector = np.array([
        np.sum(indicated_kpi_matrix[i], axis=0) +  # sum of metric values over all configurations in a row
        search_manager.min_kpi[target][i] for i in range(indicated_kpi_matrix.shape[0])])

    # search_manager.compute_kpi_functions contains a pair of kpi_metric and kpi_aggregation for each kpi target
    aggr_kpi = search_manager.compute_kpi_functions[target][1](kpi_sum_vector)

    for v in aggr_kpi:
        lp_problem += v <= target_memory


def _build_layer_to_metrics_mapping(node_to_bitwidth_indices: Dict[int, List[int]],
                                    compute_metric_fn: Callable,
                                    max_config: List[int]) -> Dict[int, Dict[int, float]]:
    """
    This function measures the sensitivity of a change in a bitwidth of a layer on the entire model.
    It builds a mapping from a node's index, to its bitwidht's effect on the model sensitivity.
    For each node and some possible node's bitwidth (according to the given search space), we use
    the framework function compute_metric_fn in order to infer
    a batch of images, and compute (using the inference results) the sensitivity metric of
    the configured mixed-precision model.

    Args:
        node_to_bitwidth_indices: Possible bitwidth indices for the different nodes.
        compute_metric_fn: Function to measure a sensitivity metric.
        max_config: A mixed-precision config which sets the maximal bitwidth candidate for each node.

    Returns:
        Mapping from each node's index in a graph, to a dictionary from the bitwidth index (of this node) to
        the sensitivity of the model.

    """

    Logger.info('Starting to evaluate metrics')
    layer_to_metrics_mapping = {}

    max_config_value = compute_metric_fn(max_config)

    for node_idx, layer_possible_bitwidths_indices in tqdm(node_to_bitwidth_indices.items(),
                                                           total=len(node_to_bitwidth_indices)):
        layer_to_metrics_mapping[node_idx] = {}

        for bitwidth_idx in layer_possible_bitwidths_indices:
            if max_config[node_idx] == bitwidth_idx:
                # This is a computation of the metric for the max configuration, assign pre-calculated value
                layer_to_metrics_mapping[node_idx][bitwidth_idx] = max_config_value
                continue

            # Create a configuration that differs at one layer only from the baseline model
            mp_model_configuration = max_config.copy()
            mp_model_configuration[node_idx] = bitwidth_idx

            # Build a distance matrix using the function we got from the framework implementation.
            layer_to_metrics_mapping[node_idx][bitwidth_idx] = compute_metric_fn(mp_model_configuration,
                                                                                 [node_idx])

    return layer_to_metrics_mapping


def _compute_kpis(node_to_bitwidth_indices: Dict[int, List[int]],
                  compute_kpi_fn: Callable,
                  min_weights_cfg: List[int],
                  min_activation_cfg: List[int]) -> Tuple[Dict[int, Dict[int, KPI]], KPI]:
    """
    This function computes and returns:
    1. The minimal possible KPI of the graph.
    2. A mapping from each node's index to a mapping from a possible bitwidth index to
    the contribution to the model's minimal KPI, if we were configuring this node with this bitwidth.

    Args:
        node_to_bitwidth_indices: Possible indices for the different nodes.
        compute_kpi_fn: Function to compute a mixed-precision model KPI for a given
        mixed-precision bitwidth configuration.
        min_weights_cfg: Mixed-Precision configuration for minimal weights precision.
        min_activation_cfg: Mixed-Precision configuration for minimal activation precision.

    Returns:
        A tuple containing a mapping from each node's index in a graph, to a dictionary from the
        bitwidth index (of this node) to the contribution to the minimal KPI of the model.
        The second element in the tuple is the minimal possible KPI.

    """

    Logger.info('Starting to compute KPIs per node and bitwidth')
    layer_to_kpi_mapping = {}

    minimal_weights_memory = compute_kpi_fn(min_weights_cfg, compute_activation_kpi=False).weights_memory
    minimal_activation_memory = compute_kpi_fn(min_activation_cfg, compute_weights_kpi=False).activation_memory
    minimal_kpi = KPI(minimal_weights_memory, minimal_activation_memory)

    for node_idx, layer_possible_bitwidths_indices in tqdm(node_to_bitwidth_indices.items(),
                                                           total=len(node_to_bitwidth_indices)):
        layer_to_kpi_mapping[node_idx] = {}
        for bitwidth_idx in layer_possible_bitwidths_indices:

            # Change the minimal KPI configuration at one node only and
            # compute this change's contribution to the model's KPI.
            weights_mp_model_configuration = min_weights_cfg.copy()
            activation_mp_model_configuration = min_activation_cfg.copy()
            weights_mp_model_configuration[node_idx] = bitwidth_idx
            activation_mp_model_configuration[node_idx] = bitwidth_idx

            weights_mp_model_kpi = compute_kpi_fn(weights_mp_model_configuration, compute_activation_kpi=False)
            activation_mp_model_kpi = compute_kpi_fn(activation_mp_model_configuration, compute_weights_kpi=False)
            contribution_to_minimal_model_weights = weights_mp_model_kpi.weights_memory - minimal_kpi.weights_memory
            contribution_to_minimal_model_activation = activation_mp_model_kpi.activation_memory - minimal_kpi.activation_memory

            layer_to_kpi_mapping[node_idx][bitwidth_idx] = KPI(
                weights_memory=contribution_to_minimal_model_weights,
                activation_memory=contribution_to_minimal_model_activation
            )

    return layer_to_kpi_mapping, minimal_kpi
