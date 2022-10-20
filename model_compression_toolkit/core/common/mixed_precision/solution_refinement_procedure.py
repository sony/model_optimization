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

from model_compression_toolkit import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import \
    MixedPrecisionSearchManager


def greedy_solution_refinement_procedure(mp_solution: List[int],
                                         search_manager: MixedPrecisionSearchManager,
                                         target_kpi: KPI) -> List[int]:
    """
    A greedy procedure to try and improve a mixed-precision solution that was found by a mixed-precision optimization algorithm.
    This procedure tries to increase the bit-width precision of configurable nodes that did not get the maximal candidate
    in the found solution.
    It iteratively goes over all such nodes, computes the KPI values on a modified configuration (with the node's next
    best candidate), filters out all configs that hold the KPI constraints and chooses one of them as an improvement step
    The choice is done in a greedy approach where we take the configuration that adds as less as possible to the KPIs.

    Args:
        mp_solution: A mixed-precision configuration that was found by a mixed-precision optimization algorithm.
        search_manager: A MixedPrecisionSearchManager object.
        target_kpi: The target KPIs for the mixed-precision search.

    Returns: A new, possibly updated, mixed-precision bit-width configuration.

    """

    new_solution = mp_solution
    changed = True

    while changed:
        changed = False
        nodes_kpis = {}
        for node_idx in range(len(mp_solution)):
            if new_solution[node_idx] == 0:
                # layer has max config in the given solution, nothing to optimize
                continue

            node_next_candidate_idx = new_solution[node_idx] - 1
            node_updated_kpis = search_manager.compute_kpi_for_config(
                config=search_manager.replace_config_in_index(new_solution, node_idx, node_next_candidate_idx))

            nodes_kpis[node_idx] = node_updated_kpis

        if len(nodes_kpis) > 0:
            # filter out new configs that don't hold the KPI restrictions
            nodes_kpis_list = [(node_idx, kpis) for node_idx, kpis in nodes_kpis.items() if target_kpi.holds_constraints(kpis)]
            sorted_by_kpi = sorted(nodes_kpis_list, key=lambda node_kpis: (node_kpis[1].total_memory,
                                                                           node_kpis[1].weights_memory,
                                                                           node_kpis[1].activation_memory))
            if len(sorted_by_kpi) > 0:
                node_to_upgrade = sorted_by_kpi[0][0]
                new_solution[node_to_upgrade] = new_solution[node_to_upgrade] - 1
                changed = True

    return new_solution
