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
from model_compression_toolkit.common.mixed_precision.kpi import KPI


def mp_integer_programming_search(layer_to_bitwidth_mapping: Dict[int, List[int]],
                                  compute_metric_fn: Callable,
                                  compute_kpi_fn: Callable,
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
        layer_to_bitwidth_mapping: Search space (mapping from each node's index to its possible bitwidth
        indices).
        compute_metric_fn: Function to compute a metric for a mixed-precision model configuration.
        compute_kpi_fn: Function to compute the KPI of the model for some mixed-precision configuration.
        target_kpi: KPI to constrain our LP problem with some resources limitations (like model' weights memory
        consumption).

    Returns:
        The mixed-precision configuration (list of indices. Each indicates the bitwidth index of a node).

    """

    # Build a mapping from each layer's index (in the model) to a dictionary that maps the
    # bitwidth index to the observed sensitivity of the model when using that bitwidth for that layer.
    layer_to_metrics_mapping = _build_layer_to_metrics_mapping(layer_to_bitwidth_mapping,
                                                               compute_metric_fn)

    # Init variables to find their values when solving the lp problem.
    layer_to_indicator_vars_mapping, layer_to_objective_vars_mapping = _init_problem_vars(layer_to_metrics_mapping)

    # Build a mapping from each node's index (in the graph) to a dictionary
    # that maps the bitwidth index to the contribution of configuring this node with this
    # bitwidth to the minimal possible KPI of the model.
    layer_to_kpi_mapping, minimal_kpi = _compute_kpis(layer_to_bitwidth_mapping,
                                                      compute_kpi_fn)

    assert minimal_kpi.weights_memory <= target_kpi.weights_memory, f'Minimal KPI cannot be greater than target KPI. Minimal KPI:{minimal_kpi}, Target KPI:{target_kpi}'
    # Add all equations and inequalities that define the problem.
    lp_problem = _formalize_problem(layer_to_indicator_vars_mapping,
                                    layer_to_metrics_mapping,
                                    layer_to_objective_vars_mapping,
                                    target_kpi,
                                    layer_to_kpi_mapping,
                                    minimal_kpi)

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
                       layer_to_kpi_mapping: Dict[int, Dict[int, KPI]],
                       minimal_kpi: KPI) -> LpProblem:
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
        layer_to_kpi_mapping: Dictionary that maps each node's index to a dictionary of bitwidth to the KPI
        contribution of the node to the minimal KPI.
        minimal_kpi: Minimal possible KPI of the graph.

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
    if target_kpi is not None and not np.isinf(target_kpi.weights_memory):
        total_weights_consumption = []
        for layer in layer_to_metrics_mapping.keys():
            weights_by_indicators = [indicator * layer_to_kpi_mapping[layer][nbits].weights_memory for nbits, indicator
                                     in layer_to_indicator_vars_mapping[layer].items()]
            total_weights_consumption.extend(weights_by_indicators)

        # Total model memory size is bounded to the given KPI.
        # Since total_weights_consumption is the contribution to the minimal possible KPI,
        # we bound the problem by the difference of the target KPI to the minimal KPI.
        lp_problem += lpSum(total_weights_consumption) <= target_kpi.weights_memory - minimal_kpi.weights_memory

    return lp_problem


def _build_layer_to_metrics_mapping(node_to_bitwidth_indices: Dict[int, List[int]],
                                    compute_metric_fn: Callable) -> Dict[int, Dict[int, float]]:
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

    Returns:
        Mapping from each node's index in a graph, to a dictionary from the bitwidth index (of this node) to
        the sensitivity of the model.

    """

    Logger.info('Starting to evaluate metrics')
    layer_to_metrics_mapping = {}

    for node_idx, layer_possible_bitwidths_indices in tqdm(node_to_bitwidth_indices.items(),
                                                           total=len(node_to_bitwidth_indices)):
        layer_to_metrics_mapping[node_idx] = {}

        for bitwidth_idx in layer_possible_bitwidths_indices:
            # Create a configuration that differs at one layer only from the baseline model
            mp_model_configuration = [0] * len(node_to_bitwidth_indices)
            mp_model_configuration[node_idx] = bitwidth_idx

            # Build a distance matrix using the function we got from the framework implementation.
            layer_to_metrics_mapping[node_idx][bitwidth_idx] = compute_metric_fn(mp_model_configuration,
                                                                                 [node_idx])

    return layer_to_metrics_mapping


def _compute_kpis(node_to_bitwidth_indices: Dict[int, List[int]],
                  compute_kpi_fn: Callable) -> Tuple[Dict[int, Dict[int, KPI]], KPI]:
    """
    This function computes and returns:
    1. The minimal possible KPI of the graph.
    2. A mapping from each node's index to a mapping from a possible bitwidth index to
    the contribution to the model's minimal KPI, if we were configuring this node with this bitwidth.

    Args:
        node_to_bitwidth_indices: Possible indices for the different nodes.
        compute_kpi_fn: Function to compute a mixed-precision model KPI for a given
        mixed-precision bitwidth configuration.

    Returns:
        A tuple containing a mapping from each node's index in a graph, to a dictionary from the
        bitwidth index (of this node) to the contribution to the minimal KPI of the model.
        The second element in the tuple is the minimal possible KPI.

    """

    Logger.info('Starting to compute KPIs per node and bitwidth')
    layer_to_kpi_mapping = {}

    # The node's candidates are sorted in a descending order, thus we take the last index of each node.
    minimal_graph_size_configuration = [node_to_bitwidth_indices[node_idx][-1] for node_idx in
                                        sorted(node_to_bitwidth_indices.keys())]

    minimal_kpi = compute_kpi_fn(minimal_graph_size_configuration)  # minimal possible kpi

    for node_idx, layer_possible_bitwidths_indices in tqdm(node_to_bitwidth_indices.items(),
                                                           total=len(node_to_bitwidth_indices)):
        layer_to_kpi_mapping[node_idx] = {}
        for bitwidth_idx in layer_possible_bitwidths_indices:

            # Change the minimal KPI configuration at one node only and
            # compute this change's contribution to the model's KPI.
            mp_model_configuration = minimal_graph_size_configuration.copy()
            mp_model_configuration[node_idx] = bitwidth_idx

            mp_model_kpi = compute_kpi_fn(mp_model_configuration)
            contribution_to_minimal_model = mp_model_kpi.weights_memory - minimal_kpi.weights_memory

            layer_to_kpi_mapping[node_idx][bitwidth_idx] = KPI(contribution_to_minimal_model)

    return layer_to_kpi_mapping, minimal_kpi
