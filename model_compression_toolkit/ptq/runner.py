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


from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from typing import Callable
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.core.common.statistics_correction.apply_bias_correction_to_graph import \
    apply_bias_correction_to_graph
from model_compression_toolkit.core.common.statistics_correction.apply_second_moment_correction_to_graph import \
    apply_second_moment_correction_to_graph
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig


def ptq_runner(tg: Graph,
               representative_data_gen: Callable,
               core_config: CoreConfig,
               fw_info: FrameworkInfo,
               fw_impl: FrameworkImplementation,
               tb_w: TensorboardWriter) -> Graph:
    """
    Quantize a graph that has final weights candidates quantization configurations.

    Args:
        tg: Graph to apply PTQ and to quantize.
        representative_data_gen (Callable): Dataset used for calibration.
        core_config: CoreConfig containing parameters of how the model should be quantized.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.)
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tb_w: A TensorBoardWriter object initialized with the logger dir path if it was set, or None otherwise.

    Returns:
        A graph after statistics correction

    """
    #############################################
    # Apply Bias Correction
    #############################################
    if core_config.quantization_config.weights_bias_correction:
        tg = apply_bias_correction_to_graph(tg,
                                            fw_impl=fw_impl)

    #############################################
    # Apply Second Moment Correction
    #############################################
    if core_config.quantization_config.weights_second_moment_correction:
        tg = apply_second_moment_correction_to_graph(tg, representative_data_gen,
                                                     core_config, fw_info, fw_impl)
    if tb_w is not None:
        tb_w.add_graph(tg, 'after_statistics_correction')

    return tg
