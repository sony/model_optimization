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


from typing import Tuple, Any

from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights

from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.common.user_info import UserInformation

from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter


def _quantize_model(tg: Graph,
                    fw_info: FrameworkInfo,
                    fw_impl: FrameworkImplementation,
                    tb_w: TensorboardWriter) -> Tuple[Any, UserInformation]:
    """
    Quantize graph's weights, and build a quantized framework model from it.

    Args:
        tg: A prepared for quantization graph.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tb_w: TensorBoardWriter object to log events.

    Returns:
        Quantized model in the input framework, and information the user may need in order to use the quantized model.
    """

    quantized_tg = quantize_graph_weights(tg,
                                          fw_info=fw_info,
                                          fw_impl=fw_impl)
    if tb_w is not None:
        tb_w.add_graph(quantized_tg, 'after_quantization')
    ######################################
    # Back2Framework
    ######################################
    # Before building a quantized model, first apply some substitutions.
    quantized_tg = substitute(quantized_tg,
                              fw_impl.get_substitutions_pre_build())

    quantized_model, user_info = fw_impl.model_builder(quantized_tg,
                                                       mode=ModelBuilderMode.QUANTIZED,
                                                       fw_info=fw_info)
    return quantized_model, user_info


def export_model(tg,
                 fw_info,
                 fw_impl,
                 tb_w,
                 bit_widths_config):

    """
    A function for quantizing the graph's weights and build a quantized framework model from it.

    Args:
        tg: A prepared for quantization graph.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tb_w: TensorBoardWriter object to log events.
        bit_widths_config: mixed-precision bit configuration to be added to model user_info

    Returns:
        Quantized model in the input framework, and information the user may need in order to use the quantized model.
    """
    quantized_model, user_info = _quantize_model(tg,
                                                 fw_info,
                                                 fw_impl,
                                                 tb_w)
    user_info.mixed_precision_cfg = bit_widths_config

    return quantized_model, user_info

