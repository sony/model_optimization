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
from typing import Dict, Tuple, Any, Optional

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization, RUTarget
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import MixedPrecisionSearchManager

# Limit ILP solver runtime in seconds
SOLVER_TIME_LIMIT = 60


def mp_integer_programming_search(search_manager: MixedPrecisionSearchManager,
                                  target_resource_utilization: ResourceUtilization = None) -> np.ndarray:
    """
    Searching and returning a mixed-precision configuration using an ILP optimization solution.
    It first builds a mapping from each layer's index (in the model) to a dictionary that maps the
    bitwidth index to the observed sensitivity of the model when using that bitwidth for that layer.
    Then, it creates a mapping from each node's index (in the graph) to a dictionary
    that maps the bitwidth index to the contribution of configuring this node with this
    bitwidth to the minimal possible resource utilization of the model.
    Then, and using these mappings, it builds an LP problem and finds an optimal solution.
    If a solution could not be found, exception is thrown.

    Args:
        search_manager: MixedPrecisionSearchManager object to be used for problem formalization.
        target_resource_utilization: Target resource utilization to constrain our LP problem with some resources limitations (like model' weights memory
        consumption).

    Returns:
        The mixed-precision configuration (1-D array of indices. Each indicates the bitwidth index of a node).

    """

    # Build a mapping from each layer's index (in the model) to a dictionary that maps the
    # bitwidth index to the observed sensitivity of the model when using that bitwidth for that layer.

    if target_resource_utilization is None or search_manager is None:
        Logger.critical("Invalid parameters: 'target_resource_utilization' and 'search_manager' must not be 'None' "
                        "for mixed-precision search. Ensure valid inputs are provided.")

    layer_to_metrics_mapping = _build_layer_to_metrics_mapping(search_manager, target_resource_utilization)

    # Init variables to find their values when solving the lp problem.
    layer_to_indicator_vars_mapping, layer_to_objective_vars_mapping = _init_problem_vars(layer_to_metrics_mapping)

    # Add all equations and inequalities that define the problem.
    lp_problem = _formalize_problem(layer_to_indicator_vars_mapping,
                                    layer_to_metrics_mapping,
                                    layer_to_objective_vars_mapping,
                                    target_resource_utilization,
                                    search_manager)

    # Use default PULP solver. Limit runtime in seconds
    solver = PULP_CBC_CMD(timeLimit=SOLVER_TIME_LIMIT)
    lp_problem.solve(solver=solver)  # Try to solve the problem.

    assert lp_problem.status == LpStatusOptimal, Logger.critical(
        "No solution was found during solving the LP problem")
    Logger.info(f"ILP status: {LpStatus[lp_problem.status]}")

    # Take the bitwidth index only if its corresponding indicator is one.
    config = np.asarray(
        [[nbits for nbits, indicator in nbits_to_indicator.items() if indicator.varValue == 1.0] for
         nbits_to_indicator
         in layer_to_indicator_vars_mapping.values()]
    ).flatten()

    if target_resource_utilization.bops_restricted():
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
                       target_resource_utilization: ResourceUtilization,
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
        target_resource_utilization: Target resource utilization to reduce our feasible solution space.
        search_manager: MixedPrecisionSearchManager object to be used for resource utilization constraints formalization.

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

    # Bound the feasible solution space with the desired resource utilization values.
    # Creates separate constraints for weights utilization and activation utilization.
    if target_resource_utilization is not None:
        indicators = []
        for layer in layer_to_metrics_mapping.keys():
            for _, indicator in layer_to_indicator_vars_mapping[layer].items():
                indicators.append(indicator)

        indicators_arr = np.array(indicators)
        indicators_matrix = np.diag(indicators_arr)

        _add_ru_constraints(search_manager=search_manager,
                            target_resource_utilization=target_resource_utilization,
                            indicators_matrix=indicators_matrix,
                            lp_problem=lp_problem,
                            non_conf_ru_dict=search_manager.non_conf_ru_dict)
    else:  # pragma: no cover
        Logger.critical("Unable to execute mixed-precision search: 'target_resource_utilization' is None. "
                        "A valid 'target_resource_utilization' is required.")
    return lp_problem


def _add_ru_constraints(search_manager: MixedPrecisionSearchManager,
                        target_resource_utilization: ResourceUtilization,
                        indicators_matrix: np.ndarray,
                        lp_problem: LpProblem,
                        non_conf_ru_dict: Dict[RUTarget, np.ndarray]):
    """
    Adding targets constraints for the Lp problem for the given target resource utilization.
    The update to the Lp problem object is done inplace.

    Args:
        search_manager:  MixedPrecisionSearchManager object to be used for resource utilization constraints formalization.
        target_resource_utilization: Target resource utilization.
        indicators_matrix: A diagonal matrix of the Lp problem's indicators.
        lp_problem: An Lp problem object to add constraint to.
        non_conf_ru_dict: A non-configurable nodes' resource utilization vectors for the constrained targets.
    """
    ru_indicated_vectors = {}
    # targets to add constraints for
    constraints_targets = target_resource_utilization.get_restricted_targets()
    # to add constraints for Total target we need to compute weight and activation
    targets_to_compute = constraints_targets
    if RUTarget.TOTAL in constraints_targets:
        targets_to_compute = targets_to_compute.union({RUTarget.ACTIVATION, RUTarget.WEIGHTS}) - {RUTarget.TOTAL}

    for target in targets_to_compute:
        ru_matrix = search_manager.compute_resource_utilization_matrix(target)    # num elements X num configurations
        indicated_ru_matrix = np.matmul(ru_matrix.T, indicators_matrix)    # num elements X num configurations

        # Sum the indicated values over all configurations, and add the value for minimal configuration once.
        # Indicated utilization values are relative to the minimal configuration, i.e. they represent the extra memory
        # that would be required if that configuration is selected).
        # Each element in a vector is an lp object representing the configurations sum term for a memory element.
        ru_vec = indicated_ru_matrix.sum(axis=1) + search_manager.min_ru[target]

        non_conf_ru_vec = non_conf_ru_dict[target]
        if non_conf_ru_vec is not None and non_conf_ru_vec.size:
            # add non-conf value as additional mem elements so that they get aggregated
            ru_vec = np.concatenate([ru_vec, non_conf_ru_vec])
        ru_indicated_vectors[target] = ru_vec

    # Add constraints only for the restricted targets in target resource utilization.
    # Adding activation constraints modifies the lp term in ru_indicated_vectors, so if both activation and total
    # are restricted we first add the constraints for total.
    if RUTarget.TOTAL in constraints_targets and RUTarget.ACTIVATION in constraints_targets:
        constraints_targets.remove(RUTarget.ACTIVATION)
        constraints_targets = list(constraints_targets) + [RUTarget.ACTIVATION]
    for target in constraints_targets:
        target_resource_utilization_value = target_resource_utilization.get_resource_utilization_dict()[target]
        aggr_ru = _aggregate_for_lp(ru_indicated_vectors, target)
        for v in aggr_ru:
            if isinstance(v, float):
                if v > target_resource_utilization_value:
                    Logger.critical(
                        f"The model cannot be quantized to meet the specified target resource utilization {target.value} "
                        f"with the value {target_resource_utilization_value}.")  # pragma: no cover
            else:
                lp_problem += v <= target_resource_utilization_value


def _aggregate_for_lp(targets_ru_vec: Dict[RUTarget, Any], target: RUTarget) -> list:
    """
    Aggregate resource utilization values for the LP.

    Args:
        targets_ru_vec: resource utilization vectors for all precomputed targets.
        target: resource utilization target.

    Returns:
        Aggregated resource utilization.
    """
    if target == RUTarget.TOTAL:
        w = lpSum(targets_ru_vec[RUTarget.WEIGHTS])
        act_ru_vec = targets_ru_vec[RUTarget.ACTIVATION]
        return [w + v for v in act_ru_vec]

    if target in [RUTarget.WEIGHTS, RUTarget.BOPS]:
        return [lpSum(targets_ru_vec[target])]

    if target == RUTarget.ACTIVATION:
        # for max aggregation, each value constitutes a separate constraint
        return list(targets_ru_vec[target])

    raise ValueError(f'Unexpected target {target}.')    # pragma: no cover


def _build_layer_to_metrics_mapping(search_manager: MixedPrecisionSearchManager,
                                    target_resource_utilization: ResourceUtilization,
                                    eps: float = EPS) -> Dict[int, Dict[int, float]]:
    """
    This function measures the sensitivity of a change in a bitwidth of a layer on the entire model.
    It builds a mapping from a node's index, to its bitwidht's effect on the model sensitivity.
    For each node and some possible node's bitwidth (according to the given search space), we use
    the framework function compute_metric_fn in order to infer
    a batch of images, and compute (using the inference results) the sensitivity metric of
    the configured mixed-precision model.

    Args:
        search_manager: MixedPrecisionSearchManager object to be used for problem formalization.
        target_resource_utilization: ResourceUtilization to constrain our LP problem with some resources limitations
        (like model' weights memory consumption).
        eps: Epsilon value to manually increase metric value (if necessary) for numerical stability

    Returns:
        Mapping from each node's index in a graph, to a dictionary from the bitwidth index (of this node) to
        the sensitivity of the model.

    """

    Logger.info('Starting to evaluate metrics')
    layer_to_metrics_mapping = {}

    if target_resource_utilization.bops_restricted():
        origin_max_config = search_manager.config_reconstruction_helper.reconstruct_config_from_virtual_graph(search_manager.max_ru_config)
        max_config_value = search_manager.compute_metric_fn(origin_max_config)
    else:
        max_config_value = search_manager.compute_metric_fn(search_manager.max_ru_config)

    for node_idx, layer_possible_bitwidths_indices in tqdm(search_manager.layer_to_bitwidth_mapping.items(),
                                                           total=len(search_manager.layer_to_bitwidth_mapping)):
        layer_to_metrics_mapping[node_idx] = {}

        for bitwidth_idx in layer_possible_bitwidths_indices:
            if search_manager.max_ru_config[node_idx] == bitwidth_idx:
                # This is a computation of the metric for the max configuration, assign pre-calculated value
                layer_to_metrics_mapping[node_idx][bitwidth_idx] = max_config_value
                continue

            # Create a configuration that differs at one layer only from the baseline model
            mp_model_configuration = search_manager.max_ru_config.copy()
            mp_model_configuration[node_idx] = bitwidth_idx

            # Build a distance matrix using the function we got from the framework implementation.
            if target_resource_utilization.bops_restricted():
                # Reconstructing original graph's configuration from virtual graph's configuration
                origin_mp_model_configuration = \
                    search_manager.config_reconstruction_helper.reconstruct_config_from_virtual_graph(
                        mp_model_configuration,
                        changed_virtual_nodes_idx=[node_idx],
                        original_base_config=origin_max_config)
                origin_changed_nodes_indices = [i for i, c in enumerate(origin_max_config) if
                                                c != origin_mp_model_configuration[i]]
                metric_value = search_manager.compute_metric_fn(
                    origin_mp_model_configuration,
                    origin_changed_nodes_indices,
                    origin_max_config)
            else:
                metric_value = search_manager.compute_metric_fn(
                    mp_model_configuration,
                    [node_idx],
                    search_manager.max_ru_config)

            layer_to_metrics_mapping[node_idx][bitwidth_idx] = max(metric_value, max_config_value + eps)

    # Finalize distance metric mapping
    search_manager.finalize_distance_metric(layer_to_metrics_mapping)

    return layer_to_metrics_mapping
