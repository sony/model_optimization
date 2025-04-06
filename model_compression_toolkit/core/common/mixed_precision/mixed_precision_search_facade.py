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

from enum import Enum
from typing import List, Callable

from model_compression_toolkit.core import MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.hessian import HessianInfoService
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import \
    MixedPrecisionSearchManager
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.solution_refinement_procedure import \
    greedy_solution_refinement_procedure


class BitWidthSearchMethod(Enum):
    INTEGER_PROGRAMMING = 0


def search_bit_width(graph: Graph,
                     fw_info: FrameworkInfo,
                     fw_impl: FrameworkImplementation,
                     target_resource_utilization: ResourceUtilization,
                     mp_config: MixedPrecisionQuantizationConfig,
                     representative_data_gen: Callable,
                     search_method: BitWidthSearchMethod = BitWidthSearchMethod.INTEGER_PROGRAMMING,
                     hessian_info_service: HessianInfoService = None) -> List[int]:
    """
    Search for an MP configuration for a given graph. Given a search_method method (by default, it's linear
    programming), we use the sensitivity_evaluator object that provides a function to compute an
    evaluation for the expected sensitivity for a bit-width configuration.
    Then, and after computing the resource utilization for each node in the graph for each bit-width in the search space,
    we search for the optimal solution, given some target_resource_utilization, the solution should fit.
    target_resource_utilization have to be passed. If it was not passed, the facade is not supposed to get here by now.

    Args:
        graph: Graph to search a MP configuration for.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
        fw_impl: FrameworkImplementation object with specific framework methods implementation.
        target_resource_utilization: Target Resource Utilization to bound our feasible solution space s.t the configuration does not violate it.
        mp_config: Mixed-precision quantization configuration.
        representative_data_gen: Dataset to use for retrieving images for the models inputs.
        search_method: BitWidthSearchMethod to define which searching method to use.
        hessian_info_service: HessianInfoService to fetch Hessian-approximation information.

    Returns:
        A MP configuration for the graph (list of integers, where the index in the list, is the node's
        index in the graph, when the graph is topology sorted, and the value in this index is the
        bit-width index on the node).

    """

    assert target_resource_utilization.is_any_restricted()

    # If we only run weights compression with MP than no need to consider activation quantization when computing the
    # MP metric (it adds noise to the computation)
    tru = target_resource_utilization
    weight_only_restricted = tru.weight_restricted() and not (tru.activation_restricted() or
                                                              tru.total_mem_restricted() or
                                                              tru.bops_restricted())
    disable_activation_for_metric = weight_only_restricted or not graph.has_any_configurable_activation()

    # Set Sensitivity Evaluator for MP search. It should always work with the original MP graph,
    # even if a virtual graph was created (and is used only for BOPS utilization computation purposes)
    se = fw_impl.get_sensitivity_evaluator(
        graph,
        mp_config,
        representative_data_gen=representative_data_gen,
        fw_info=fw_info,
        disable_activation_for_metric=disable_activation_for_metric,
        hessian_info_service=hessian_info_service)

    if search_method != BitWidthSearchMethod.INTEGER_PROGRAMMING:
        raise NotImplementedError()

    # Validation is skipped during the mixed-precision search configuration because fusing information is not
    # relevant for the virtual graph. Therefore, validation checks are disabled before the search begins and
    # re-enabled once it completes.
    graph.skip_validation_check = True

    # Search manager and LP are highly coupled, so LP search method was moved inside search manager.
    search_manager = MixedPrecisionSearchManager(graph,
                                                 fw_info,
                                                 fw_impl,
                                                 se,
                                                 target_resource_utilization)
    result_bit_cfg = search_manager.search()

    graph.skip_validation_check = False

    if mp_config.refine_mp_solution:
        result_bit_cfg = greedy_solution_refinement_procedure(result_bit_cfg, search_manager, target_resource_utilization)

    return result_bit_cfg
