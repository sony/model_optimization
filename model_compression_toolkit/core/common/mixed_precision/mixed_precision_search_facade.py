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

import copy
from enum import Enum
import numpy as np
from typing import List, Callable, Dict

from model_compression_toolkit.core import MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import HessianInfoService
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization, RUTarget
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.ru_functions_mapping import ru_functions_mapping
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import MixedPrecisionSearchManager
from model_compression_toolkit.core.common.mixed_precision.search_methods.linear_programming import \
    mp_integer_programming_search
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.solution_refinement_procedure import \
    greedy_solution_refinement_procedure
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.logger import Logger


class BitWidthSearchMethod(Enum):
    # When adding a new search_methods MP configuration method, these enum and factory dictionary
    # should be updated with it's kind and a search_method implementation.
    INTEGER_PROGRAMMING = 0


search_methods = {
    BitWidthSearchMethod.INTEGER_PROGRAMMING: mp_integer_programming_search}


def search_bit_width(graph_to_search_cfg: Graph,
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
        graph_to_search_cfg: Graph to search a MP configuration for.
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

    # target_resource_utilization have to be passed. If it was not passed, the facade is not supposed to get here by now.
    if target_resource_utilization is None:
        Logger.critical("Target ResourceUtilization is required for the bit-width search method's configuration.")  # pragma: no cover

    # Set graph for MP search
    graph = copy.deepcopy(graph_to_search_cfg)  # Copy graph before searching
    if target_resource_utilization.bops < np.inf:
        # Since Bit-operations count target resource utilization is set, we need to reconstruct the graph for the MP search
        graph = substitute(graph, fw_impl.get_substitutions_virtual_weights_activation_coupling())

    # If we only run weights compression with MP than no need to consider activation quantization when computing the
    # MP metric (it adds noise to the computation)
    disable_activation_for_metric = (target_resource_utilization.weights_memory < np.inf and
                                    (target_resource_utilization.activation_memory == np.inf and
                                     target_resource_utilization.total_memory == np.inf and
                                     target_resource_utilization.bops == np.inf)) or graph_to_search_cfg.is_single_activation_cfg()

    # Set Sensitivity Evaluator for MP search. It should always work with the original MP graph,
    # even if a virtual graph was created (and is used only for BOPS utilization computation purposes)
    se = fw_impl.get_sensitivity_evaluator(
        graph_to_search_cfg,
        mp_config,
        representative_data_gen=representative_data_gen,
        fw_info=fw_info,
        disable_activation_for_metric=disable_activation_for_metric,
        hessian_info_service=hessian_info_service)

    # Each pair of (resource utilization method, resource utilization aggregation) should match to a specific
    # provided target resource utilization
    ru_functions = ru_functions_mapping

    # Instantiate a manager object
    search_manager = MixedPrecisionSearchManager(graph,
                                                 fw_info,
                                                 fw_impl,
                                                 se,
                                                 ru_functions,
                                                 target_resource_utilization,
                                                 original_graph=graph_to_search_cfg)

    if search_method in search_methods:  # Get a specific search function
        search_method_fn = search_methods.get(search_method)
    else:
        raise NotImplemented  # pragma: no cover

    # Search for the desired mixed-precision configuration
    result_bit_cfg = search_method_fn(search_manager,
                                      target_resource_utilization)

    if mp_config.refine_mp_solution:
        result_bit_cfg = greedy_solution_refinement_procedure(result_bit_cfg, search_manager, target_resource_utilization)

    return result_bit_cfg
