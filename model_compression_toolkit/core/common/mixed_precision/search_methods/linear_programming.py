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
from typing import Dict, Tuple, Any

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization, RUTarget
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import MixedPrecisionSearchManager

# Limit ILP solver runtime in seconds
SOLVER_TIME_LIMIT = 60


def mp_integer_programming_search(search_manager: MixedPrecisionSearchManager,
                                  target_resource_utilization: ResourceUtilization) -> List[int]:
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
        The mixed-precision configuration (A list of indices. Each indicates the bitwidth index of a node).

    """

    # Build a mapping from each layer's index (in the model) to a dictionary that maps the
    # bitwidth index to the observed sensitivity of the model when using that bitwidth for that layer.

    layer_to_sensitivity_mapping = search_manager.build_sensitivity_mapping()

    # Init variables to find their values when solving the lp problem.
    layer_to_indicator_vars_mapping, layer_to_objective_vars_mapping = _init_problem_vars(layer_to_sensitivity_mapping)

    # Add all equations and inequalities that define the problem.
    lp_problem = _formalize_problem(layer_to_indicator_vars_mapping,
                                    layer_to_sensitivity_mapping,
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

    return config.tolist()


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
    assert target_resource_utilization and target_resource_utilization.is_any_restricted()

    indicators = []
    for layer in layer_to_metrics_mapping.keys():
        for _, indicator in layer_to_indicator_vars_mapping[layer].items():
            indicators.append(indicator)

    indicators_vec = np.array(indicators)

    _add_ru_constraints(search_manager=search_manager,
                        target_resource_utilization=target_resource_utilization,
                        indicators_vec=indicators_vec,
                        lp_problem=lp_problem)
    return lp_problem


def _add_ru_constraints(search_manager: MixedPrecisionSearchManager,
                        target_resource_utilization: ResourceUtilization,
                        indicators_vec: np.ndarray,
                        lp_problem: LpProblem):
    """
    Adding targets constraints for the Lp problem for the given target resource utilization.
    The update to the Lp problem object is done inplace.

    Args:
        search_manager:  MixedPrecisionSearchManager object to be used for resource utilization constraints formalization.
        target_resource_utilization: Target resource utilization.
        indicators_vec: A vector of the Lp problem's indicators.
        lp_problem: An Lp problem object to add constraint to.
    """
    candidates_ru = search_manager.compute_resource_utilization_matrices()
    min_ru = search_manager.min_ru
    target_ru = target_resource_utilization.get_resource_utilization_dict(restricted_only=True)
    assert candidates_ru.keys() == target_ru.keys()

    for target, ru_matrix in candidates_ru.items():
        # We expect 2d matrix of shape (num candidates, m). For cumulative metrics (weights, bops) m=1 - overall
        # utilization. For max metrics (activation, total) m=num memory elements (max element depends on configuration)
        assert ru_matrix.ndim == 2
        if target in [RUTarget.WEIGHTS, RUTarget.BOPS]:
            assert ru_matrix.shape[1] == 1

        # ru values are relative to the minimal configuration, so we adjust the target ru accordingly
        ru_constraint = target_ru[target] - min_ru[target]
        if any(ru_constraint < 0):
            raise ValueError(f"The model cannot be quantized to meet the specified target resource utilization "
                             f"{target.value} with the value {target_ru[target]}.")

        indicated_ru_matrix = ru_matrix.T * indicators_vec
        # build lp sum term over all candidates
        ru_vec = indicated_ru_matrix.sum(axis=1)

        # For cumulative metrics a single constraint is added, for max metrics a separate constraint
        # is added for each memory element (each element < target => max element < target).
        assert len(ru_vec) == len(ru_constraint)
        for v, c in zip(ru_vec, ru_constraint):
            lp_problem += v <= c
