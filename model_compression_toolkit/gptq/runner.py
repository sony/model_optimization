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

from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.statistics_correction.statistics_correction import \
    apply_statistics_correction
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfigV2
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.gptq.common.gptq_training import gptq_training

from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.core.common.statistics_correction.apply_bias_correction_to_graph import \
    apply_bias_correction_to_graph
from model_compression_toolkit.logger import Logger


def _apply_gptq(gptq_config: GradientPTQConfigV2,
                representative_data_gen: Callable,
                tb_w: TensorboardWriter,
                tg: Graph,
                tg_bias: Graph,
                fw_info: FrameworkInfo,
                fw_impl: FrameworkImplementation) -> Graph:
    """
    Apply GPTQ to improve accuracy of quantized model.
    Build two models from a graph: A teacher network (float model) and a student network (quantized model).
    and use the dataset generator to pass images through the teacher and student networks to get intermediate
    layers outputs and maximize their similarity.

    Args:
        gptq_config: Configuration for using GPTQ (e.g. optimizer).
        representative_data_gen: Dataset used for calibration.
        tb_w: TensorBoardWriter object to log events.
        tg: Float Reference Graph.
        tg_bias: Graph of quantized model.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).
        fw_impl: Framework implementation per framework
    Returns:

    """
    if gptq_config is not None and gptq_config.n_epochs > 0:
        Logger.info("Using experimental Gradient Based PTQ: If you encounter an issue "
                           "please file a bug. To disable it, do not pass a gptq configuration.")

        tg_bias = gptq_training(tg,
                                tg_bias,
                                gptq_config,
                                representative_data_gen,
                                fw_impl,
                                fw_info)

        if tb_w is not None:
            tb_w.add_graph(tg_bias, 'after_gptq')
    return tg_bias


def gptq_runner(tg: Graph,
                core_config: CoreConfig,
                gptq_config: GradientPTQConfigV2,
                representative_data_gen: Callable,
                gptq_representative_data_gen: Callable,
                fw_info: FrameworkInfo,
                fw_impl: FrameworkImplementation,
                tb_w: TensorboardWriter) -> Graph:
    """
    Quantize a graph that has final weights candidates quantization configurations.
    Before we quantize the graph weights, we apply GPTQ to get an improved graph.

    Args:
        tg: Graph to apply GPTQ and to quantize.
        core_config: CoreConfig containing parameters of how the model should be quantized.
        gptq_config: GradientPTQConfig with parameters about the tuning process.
        representative_data_gen: Dataset used for calibration.
        gptq_representative_data_gen: Dataset used for GPTQ training
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.)
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tb_w: A TensorBoardWriter object initialized with the logger dir path if it was set, or None otherwise.

    Returns:
        A graph after model weights GPTQ fine-tuning.

    """

    #############################################
    # Apply Statistics Correction
    #############################################
    tg_bias = apply_statistics_correction(tg, representative_data_gen, core_config, fw_info, fw_impl, tb_w)

    if tb_w is not None:
        tb_w.add_graph(tg_bias, 'after_bias_correction')
    #############################################
    # Gradient Based Post Training Quantization
    #############################################
    tg_gptq = _apply_gptq(gptq_config,
                          gptq_representative_data_gen,
                          tb_w,
                          tg,
                          tg_bias,
                          fw_info,
                          fw_impl)

    return tg_gptq
