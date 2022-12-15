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

import numpy as np
from pulp import *
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI, KPITarget
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import MixedPrecisionSearchManager


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

    if target_kpi is None or search_manager is None:
        Logger.critical("Can't run mixed precision search with given target_kpi=None or search_manager=None."
                        "Please provide a valid target_kpi and check the mixed precision parameters values.")

    layer_to_metrics_mapping = _build_layer_to_metrics_mapping(search_manager, target_kpi)

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

    if target_kpi.bops < np.inf:
        return search_manager.config_reconstruction_helper.reconstruct_config_from_virtual_graph(config)
    else:
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

        for target, kpi_value in target_kpi.get_kpi_dict().items():
            if not np.isinf(kpi_value):
                non_conf_kpi_vector = None if search_manager.non_conf_kpi_dict is None \
                    else search_manager.non_conf_kpi_dict.get(target)
                _add_set_of_kpi_constraints(search_manager=search_manager,
                                            target=target,
                                            target_kpi_value=kpi_value,
                                            indicators_matrix=indicators_matrix,
                                            lp_problem=lp_problem,
                                            non_conf_kpi_vector=non_conf_kpi_vector)
    else:  # pragma: no cover
        raise Logger.critical("Can't run mixed-precision search with given target_kpi=None."
                              "Please provide a valid target_kpi.")
    return lp_problem


def _add_set_of_kpi_constraints(search_manager: MixedPrecisionSearchManager,
                                target: KPITarget,
                                target_kpi_value: float,
                                indicators_matrix: np.ndarray,
                                lp_problem: LpProblem,
                                non_conf_kpi_vector: np.ndarray):
    """
    Adding a constraint for the Lp problem for the given KPI target.
    The update to the Lp problem object is done inplace.

    Args:
        search_manager:  MixedPrecisionSearchManager object to be used for kpi constraints formalization.
        target: A KPITarget.
        target_kpi_value: Target KPI value of the given KPI target for which the constraint is added.
        indicators_matrix: A diagonal matrix of the Lp problem's indicators.
        lp_problem: An Lp problem object to add constraint to.
        non_conf_kpi_vector: A non-configurable nodes' KPI vector.

    """

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
    # get aggregated KPI, considering both configurable and non-configurable nodes
    if non_conf_kpi_vector is None or len(non_conf_kpi_vector) == 0:
        aggr_kpi = search_manager.compute_kpi_functions[target][1](kpi_sum_vector)
    else:
        aggr_kpi = search_manager.compute_kpi_functions[target][1](np.concatenate([kpi_sum_vector, non_conf_kpi_vector]))

    for v in aggr_kpi:
        if isinstance(v, float):
            if v > target_kpi_value:
                Logger.critical(f"The model can't be quantized to satisfy target KPI {target.value} with value {target_kpi_value}")  # pragma: no cover
        else:
            lp_problem += v <= target_kpi_value


def _build_layer_to_metrics_mapping(search_manager: MixedPrecisionSearchManager,
                                    target_kpi: KPI) -> Dict[int, Dict[int, float]]:
    """
    This function measures the sensitivity of a change in a bitwidth of a layer on the entire model.
    It builds a mapping from a node's index, to its bitwidht's effect on the model sensitivity.
    For each node and some possible node's bitwidth (according to the given search space), we use
    the framework function compute_metric_fn in order to infer
    a batch of images, and compute (using the inference results) the sensitivity metric of
    the configured mixed-precision model.

    Args:
        search_manager: MixedPrecisionSearchManager object to be used for problem formalization.
        target_kpi: KPI to constrain our LP problem with some resources limitations (like model' weights memory
        consumption).

    Returns:
        Mapping from each node's index in a graph, to a dictionary from the bitwidth index (of this node) to
        the sensitivity of the model.

    """

    Logger.info('Starting to evaluate metrics')
    layer_to_metrics_mapping = {}

    is_bops_target_kpi = target_kpi.bops < np.inf

    if is_bops_target_kpi:
        origin_max_config = search_manager.config_reconstruction_helper.reconstruct_config_from_virtual_graph(search_manager.max_kpi_config)
        max_config_value = search_manager.compute_metric_fn(origin_max_config)
    else:
        max_config_value = search_manager.compute_metric_fn(search_manager.max_kpi_config)

    for node_idx, layer_possible_bitwidths_indices in tqdm(search_manager.layer_to_bitwidth_mapping.items(),
                                                           total=len(search_manager.layer_to_bitwidth_mapping)):
        layer_to_metrics_mapping[node_idx] = {}

        for bitwidth_idx in layer_possible_bitwidths_indices:
            if search_manager.max_kpi_config[node_idx] == bitwidth_idx:
                # This is a computation of the metric for the max configuration, assign pre-calculated value
                layer_to_metrics_mapping[node_idx][bitwidth_idx] = max_config_value
                continue

            # Create a configuration that differs at one layer only from the baseline model
            mp_model_configuration = search_manager.max_kpi_config.copy()
            mp_model_configuration[node_idx] = bitwidth_idx

            # Build a distance matrix using the function we got from the framework implementation.
            if is_bops_target_kpi:
                # Reconstructing original graph's configuration from virtual graph's configuration
                origin_mp_model_configuration = \
                    search_manager.config_reconstruction_helper.reconstruct_config_from_virtual_graph(
                        mp_model_configuration,
                        changed_virtual_nodes_idx=[node_idx],
                        original_base_config=origin_max_config)
                origin_changed_nodes_indices = [i for i, c in enumerate(origin_max_config) if
                                                c != origin_mp_model_configuration[i]]
                layer_to_metrics_mapping[node_idx][bitwidth_idx] = search_manager.compute_metric_fn(
                    origin_mp_model_configuration,
                    origin_changed_nodes_indices,
                    origin_max_config)
            else:
                layer_to_metrics_mapping[node_idx][bitwidth_idx] = search_manager.compute_metric_fn(
                    mp_model_configuration,
                    [node_idx],
                    search_manager.max_kpi_config)

    return layer_to_metrics_mapping
