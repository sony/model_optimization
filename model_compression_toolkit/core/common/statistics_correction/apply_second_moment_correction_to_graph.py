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
from typing import Callable

from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute


def apply_second_moment_correction_to_graph(graph_to_apply_second_moment_correction: Graph,
                                            representative_data_gen: Callable,
                                            core_config: CoreConfig,
                                            fw_info: FrameworkInfo,
                                            fw_impl: FrameworkImplementation) -> Graph:
    """
     Apply second moment correction on graph.
     Args:
        graph_to_apply_second_moment_correction: Graph to apply second moment correction.
        representative_data_gen (Callable): Dataset used for calibration.
        core_config (CoreConfig): Configuration object containing parameters of how the model should be
         quantized, including mixed precision parameters.
        fw_info: FrameworkInfo object with information about the specific framework's model.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

     Returns:
         Graph after second moment correction.
     """
    graph = copy.deepcopy(graph_to_apply_second_moment_correction)
    semi_quantized_model = fw_impl.quantized_model_builder_for_second_moment_correction(graph, fw_info)
    fw_impl.apply_second_moment_correction(semi_quantized_model, core_config, representative_data_gen, graph)
    graph = substitute(graph, fw_impl.get_substitutions_after_second_moment_correction(core_config.quantization_config))
    return graph

