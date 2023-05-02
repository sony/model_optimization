# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import List

from model_compression_toolkit.core import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import \
    MixedPrecisionSearchManager
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.logger import Logger
import numpy as np


def greedy_solution_refinement_procedure(mp_solution: List[int],
                                         search_manager: MixedPrecisionSearchManager,
                                         target_kpi: KPI) -> List[int]:
    """
    A greedy procedure to try and improve a mixed-precision solution that was found by a mixed-precision optimization
    algorithm.
    This procedure tries to increase the bit-width precision of configurable nodes that did not get the maximal
    candidate
    in the found solution.
    It iteratively goes over all such nodes, computes the KPI values on a modified configuration (with the node's next
    best candidate), filters out all configs that hold the KPI constraints and chooses one of them as an improvement
    step
    The choice is done in a greedy approach where we take the configuration that modifies the KPI the least.

    Args:
        mp_solution: A mixed-precision configuration that was found by a mixed-precision optimization algorithm.
        search_manager: A MixedPrecisionSearchManager object.
        target_kpi: The target KPIs for the mixed-precision search.

    Returns: A new, possibly updated, mixed-precision bit-width configuration.

    """
    # Refinement is not supported for BOPs KPI for now...
    if target_kpi.bops < np.inf:
        Logger.info(f'Target KPI constraint BOPs - Skipping MP greedy solution refinement')
        return mp_solution

    new_solution = mp_solution.copy()
    changed = True

    while changed:
        changed = False
        nodes_kpis = {}
        nodes_next_candidate = {}

        for node_idx in range(len(mp_solution)):
            if new_solution[node_idx] == 0:
                # layer has max config in the given solution, nothing to optimize
                continue

            node_candidates = search_manager.graph.get_configurable_sorted_nodes()[node_idx].candidates_quantization_cfg
            valid_candidates = _get_valid_candidates_indices(node_candidates, new_solution[node_idx])

            # Create a list of KPIs for the valid candidates.
            updated_kpis = []
            for valid_idx in valid_candidates:
                node_updated_kpis = search_manager.compute_kpi_for_config(
                    config=search_manager.replace_config_in_index(new_solution, node_idx, valid_idx))
                updated_kpis.append(node_updated_kpis)

            # filter out new configs that don't hold the KPI restrictions
            node_filtered_kpis = [(node_idx, kpis) for node_idx, kpis in zip(valid_candidates,updated_kpis) if
                               target_kpi.holds_constraints(kpis)]

            if len(node_filtered_kpis) > 0:
                sorted_by_kpi = sorted(node_filtered_kpis, key=lambda node_kpis: (node_kpis[1].total_memory,
                                                                               node_kpis[1].weights_memory,
                                                                               node_kpis[1].activation_memory))
                nodes_kpis[node_idx] = sorted_by_kpi[0][1]
                nodes_next_candidate[node_idx] = sorted_by_kpi[0][0]


        if len(nodes_kpis) > 0:
            # filter out new configs that don't hold the KPI restrictions
            node_filtered_kpis = [(node_idx, kpis) for node_idx, kpis in nodes_kpis.items()]
            sorted_by_kpi = sorted(node_filtered_kpis, key=lambda node_kpis: (node_kpis[1].total_memory,
                                                                           node_kpis[1].weights_memory,
                                                                           node_kpis[1].activation_memory))

            node_idx_to_upgrade = sorted_by_kpi[0][0]
            new_solution[node_idx_to_upgrade] = nodes_next_candidate[node_idx_to_upgrade]
            changed = True

    Logger.info(f'Greedy MP algorithm changed configuration from: {mp_solution} to {new_solution}')
    return new_solution


def _get_valid_candidates_indices(node_candidates: List[CandidateNodeQuantizationConfig],
                                  current_chosen_index: int) -> List[int]:
    """
    Find node's valid candidates to try and improve the node's MP chosen candidate.
    Valid indices are indices of candidates that have higher number of bits for both weights and activations.

    Args:
        node_candidates: Candidates of the node.
        current_chosen_index: Current index in MP configuration of the node.

    Returns:
        List of indices of valid candidates.
    """

    current_candidate = node_candidates[current_chosen_index]
    weights_num_bits = current_candidate.weights_quantization_cfg.weights_n_bits
    activation_num_bits = current_candidate.activation_quantization_cfg.activation_n_bits

    # Filter candidates that have higher bit-width for both weights and activations (except for the current index).
    return [i for i, c in enumerate(node_candidates) if c.activation_quantization_cfg.activation_n_bits >= activation_num_bits and c.weights_quantization_cfg.weights_n_bits >= weights_num_bits and not (c.activation_quantization_cfg.activation_n_bits == activation_num_bits and c.weights_quantization_cfg.weights_n_bits == weights_num_bits)]
