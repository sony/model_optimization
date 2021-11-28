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

from typing import Callable, List

from model_compression_toolkit.common import Graph, Logger
from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.mixed_precision.mixed_precision_search_manager import MixedPrecisionSearchManager
from model_compression_toolkit.common.mixed_precision.search_methods.linear_programming import \
    mp_integer_programming_search
from model_compression_toolkit.common.framework_info import FrameworkInfo

# When adding a new search_methods MP configuration method, these enum and factory dictionary
# should be updated with it's kind and a search_method implementation.
class BitWidthSearchMethod(Enum):
    INTEGER_PROGRAMMING = 0


search_methods = {
    BitWidthSearchMethod.INTEGER_PROGRAMMING: mp_integer_programming_search}


def search_bit_width(graph_to_search_cfg: Graph,
                     qc: MixedPrecisionQuantizationConfig,
                     fw_info: FrameworkInfo,
                     target_kpi: KPI,
                     get_sensitivity_evaluation: Callable = None,
                     search_method: BitWidthSearchMethod = BitWidthSearchMethod.INTEGER_PROGRAMMING) -> List[int]:
    """
    Search for a MP configuration for a given graph. Given a search_method method (by default, it's linear
    programming), we use the get_sensitivity_evaluation function to get a function to compute an
    evaluation for the expected sensitivity for a bitwidth configuration.
    Then, and after computing the KPI for each node in the graph for each bitwidth in the search space,
    we search for the optimal solution, given some target_kpi, the solution should fit.
    target_kpi have to be passed. If it was not passed, the facade is not supposed to get here by now.

    Args:
        graph_to_search_cfg: Graph to search a MP configuration for.
        qc: MixedPrecisionQuantizationConfig the graph was prepared according to.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
        target_kpi: Target KPI to bound our feasible solution space s.t the configuration does not violates it.
        get_sensitivity_evaluation: Function specific to the model's framework, which builds and returns
        a function that evaluates the sensitivity of a bitwidth configuration for the MP model.
        search_method: BitWidthSearchMethod to define which searching method to use.

    Returns:
        A MP configuration for the graph (list of integers, where the index in the list, is the node's
        index in the graph, when the graph is topology sorted, and the value in this index is the
        bitwidth index on the node).

    """

    # target_kpi have to be passed. If it was not passed, the facade is not supposed to get here by now.
    if target_kpi is None:
        Logger.critical('Target KPI have to be passed for search_methods bitwidth configuration')

    graph = copy.deepcopy(graph_to_search_cfg)  # Copy graph before searching

    # Instantiate a manager object
    search_manager = MixedPrecisionSearchManager(graph,
                                                 qc,
                                                 fw_info,
                                                 get_sensitivity_evaluation)

    if search_method in search_methods:  # Get a specific search function
        search_method_fn = search_methods.get(search_method)
    else:
        raise NotImplemented

    # Search for the desired mixed-precision configuration
    result_bit_cfg = search_method_fn(search_manager.layer_to_bitwidth_mapping,
                                      search_manager.compute_metric_fn,
                                      search_manager.get_kpi_metric(),
                                      target_kpi)

    return result_bit_cfg
