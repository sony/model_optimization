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

from model_compression_toolkit import MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.common import Graph, Logger
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI, KPITarget
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_functions_mapping import kpi_functions_mapping
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import MixedPrecisionSearchManager
from model_compression_toolkit.core.common.mixed_precision.search_methods.linear_programming import \
    mp_integer_programming_search
from model_compression_toolkit.core.common.framework_info import FrameworkInfo


# When adding a new search_methods MP configuration method, these enum and factory dictionary
# should be updated with it's kind and a search_method implementation.
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import SensitivityEvaluation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute


class BitWidthSearchMethod(Enum):
    INTEGER_PROGRAMMING = 0


search_methods = {
    BitWidthSearchMethod.INTEGER_PROGRAMMING: mp_integer_programming_search}


def search_bit_width(graph_to_search_cfg: Graph,
                     fw_info: FrameworkInfo,
                     fw_impl: FrameworkImplementation,
                     target_kpi: KPI,
                     mp_config: MixedPrecisionQuantizationConfigV2,
                     representative_data_gen: Callable,
                     search_method: BitWidthSearchMethod = BitWidthSearchMethod.INTEGER_PROGRAMMING) -> List[int]:
    """
    Search for an MP configuration for a given graph. Given a search_method method (by default, it's linear
    programming), we use the sensitivity_evaluator object that provides a function to compute an
    evaluation for the expected sensitivity for a bit-width configuration.
    Then, and after computing the KPI for each node in the graph for each bit-width in the search space,
    we search for the optimal solution, given some target_kpi, the solution should fit.
    target_kpi have to be passed. If it was not passed, the facade is not supposed to get here by now.

    Args:
        graph_to_search_cfg: Graph to search a MP configuration for.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
        fw_impl: FrameworkImplementation object with specific framework methods implementation.
        target_kpi: Target KPI to bound our feasible solution space s.t the configuration does not violate it.
        mp_config: Mixed-precision quantization configuration.
        representative_data_gen: Dataset to use for retrieving images for the models inputs.
        search_method: BitWidthSearchMethod to define which searching method to use.

    Returns:
        A MP configuration for the graph (list of integers, where the index in the list, is the node's
        index in the graph, when the graph is topology sorted, and the value in this index is the
        bit-width index on the node).

    """

    # target_kpi have to be passed. If it was not passed, the facade is not supposed to get here by now.
    if target_kpi is None:
        Logger.critical('Target KPI have to be passed for search_methods bit-width configuration')

    # Set graph for MP search
    graph = copy.deepcopy(graph_to_search_cfg)  # Copy graph before searching
    if target_kpi.bops < np.inf:
        # Since Bit-operations count target KPI is set, we need to reconstruct the graph for the MP search
        graph = substitute(graph, fw_impl.get_substitutions_virtual_weights_activation_coupling())

    # Set Sensitivity Evaluator for MP search. It should always work with the original MP graph,
    # even if a virtual graph was created (and is used only for BOPS KPI computation purposes)
    se = fw_impl.get_sensitivity_evaluator(
        graph_to_search_cfg,
        mp_config,
        representative_data_gen=representative_data_gen,
        fw_info=fw_info)

    # Each pair of (KPI method, KPI aggregation) should match to a specific provided kpi target
    kpi_functions = kpi_functions_mapping

    # Compute non-configurable nodes KPIs for each KPI target
    non_conf_kpi_dict = _non_configurable_nodes_kpi(kpi_functions, target_kpi, graph, fw_info, fw_impl)

    # Instantiate a manager object
    search_manager = MixedPrecisionSearchManager(graph,
                                                 fw_info,
                                                 fw_impl,
                                                 se,
                                                 kpi_functions,
                                                 original_graph=graph_to_search_cfg)

    if search_method in search_methods:  # Get a specific search function
        search_method_fn = search_methods.get(search_method)
    else:
        raise NotImplemented

    # Search for the desired mixed-precision configuration
    result_bit_cfg = search_method_fn(search_manager,
                                      target_kpi,
                                      non_conf_kpi_dict)

    return result_bit_cfg


def _non_configurable_nodes_kpi(kpi_functions, target_kpi, graph, fw_info, fw_impl) -> Dict[KPITarget, np.ndarray]:
    """
    Computes a KPI vector of all non-configurable nodes in the given graph for each of the KPI target.

    Args:
        kpi_functions: A dictionary with pairs of (MpKpiMethod, MpKpiAggregationMethod) mapping a KPITarget to
                a couple of kpi metric function and kpi aggregation function.
        target_kpi: Target KPI to bound our feasible solution space s.t the configuration does not violate it.
        graph: Graph to search a MP configuration for.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
        fw_impl: FrameworkImplementation object with specific framework methods implementation.

    Returns: A mapping between a KPITarget and its non-configurable nodes' KPI vector.

    """

    non_conf_kpi_dict = {}
    for target, kpi_value in target_kpi.get_kpi_dict().items():
        if not np.isinf(kpi_value):
            # Call for the KPI method of the given target - empty quantization configuration list is passed since we
            # compute for non-configurable nodes
            kpi_vector = kpi_functions[target][0]([], graph, fw_info, fw_impl)

            non_conf_kpi_dict[target] = kpi_vector

    return non_conf_kpi_dict
