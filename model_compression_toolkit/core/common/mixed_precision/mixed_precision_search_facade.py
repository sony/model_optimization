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

import copy
from enum import Enum

from typing import List

from model_compression_toolkit.core.common import Graph, Logger
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_functions_mapping import kpi_functions_mapping
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import MixedPrecisionSearchManager
from model_compression_toolkit.core.common.mixed_precision.search_methods.linear_programming import \
    mp_integer_programming_search
from model_compression_toolkit.core.common.framework_info import FrameworkInfo


# When adding a new search_methods MP configuration method, these enum and factory dictionary
# should be updated with it's kind and a search_method implementation.
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import SensitivityEvaluation


class BitWidthSearchMethod(Enum):
    INTEGER_PROGRAMMING = 0


search_methods = {
    BitWidthSearchMethod.INTEGER_PROGRAMMING: mp_integer_programming_search}


def search_bit_width(graph_to_search_cfg: Graph,
                     fw_info: FrameworkInfo,
                     target_kpi: KPI,
                     sensitivity_evaluator: SensitivityEvaluation = None,
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
        target_kpi: Target KPI to bound our feasible solution space s.t the configuration does not violate it.
        sensitivity_evaluator: A SensitivityEvaluation which provides a function that evaluates the sensitivity of
            a bit-width configuration for the MP model.
        search_method: BitWidthSearchMethod to define which searching method to use.

    Returns:
        A MP configuration for the graph (list of integers, where the index in the list, is the node's
        index in the graph, when the graph is topology sorted, and the value in this index is the
        bit-width index on the node).

    """

    # target_kpi have to be passed. If it was not passed, the facade is not supposed to get here by now.
    if target_kpi is None:
        Logger.critical('Target KPI have to be passed for search_methods bit-width configuration')

    graph = copy.deepcopy(graph_to_search_cfg)  # Copy graph before searching

    # Each pair of (KPI method, KPI aggregation) should match to a specific provided kpi target
    kpi_functions = kpi_functions_mapping

    # Instantiate a manager object
    search_manager = MixedPrecisionSearchManager(graph,
                                                 fw_info,
                                                 sensitivity_evaluator,
                                                 kpi_functions)

    if search_method in search_methods:  # Get a specific search function
        search_method_fn = search_methods.get(search_method)
    else:
        raise NotImplemented

    # Search for the desired mixed-precision configuration
    result_bit_cfg = search_method_fn(search_manager,
                                      target_kpi)

    return result_bit_cfg
