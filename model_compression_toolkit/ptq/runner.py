# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.core.common.bias_correction.apply_bias_correction_to_graph import \
    apply_bias_correction_to_graph


def ptq_runner(tg: Graph,
               fw_info: FrameworkInfo,
               fw_impl: FrameworkImplementation,
               tb_w: TensorboardWriter) -> Graph:
    """
    Quantize a graph that has final weights candidates quantization configurations.

    Args:
        tg: Graph to apply GPTQ and to quantize.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.)
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tb_w: A TensorBoardWriter object initialized with the logger dir path if it was set, or None otherwise.

    Returns:
        A graph after bias correction

    """
    #############################################
    # Apply Bias Correction
    #############################################
    tg_bias = apply_bias_correction_to_graph(tg,
                                             fw_impl=fw_impl)
    if tb_w is not None:
        tb_w.add_graph(tg_bias, 'after_bias_correction')

    return tg_bias
