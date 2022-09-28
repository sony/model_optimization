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


from typing import Callable

from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.statistics_correction.statistics_correction import \
    apply_statistics_correction
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter


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
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tb_w: A TensorBoardWriter object initialized with the logger dir path if it was set, or None otherwise.

    Returns:
        A graph after statistics correction

    """
    #############################################
    # Statistics Correction
    #############################################
    tg = apply_statistics_correction(tg, representative_data_gen, core_config, fw_info, fw_impl, tb_w)
    return tg
