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
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.statistics_correction.apply_bias_correction_to_graph import \
    apply_bias_correction_to_graph
from model_compression_toolkit.core.common.statistics_correction.apply_second_moment_correction_to_graph import \
    apply_second_moment_correction_to_graph
from model_compression_toolkit.core.common.statistics_correction.compute_bias_correction_of_graph import \
    compute_bias_correction_of_graph
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter


def statistics_correction_runner(transformed_graph: Graph,
                                 core_config: CoreConfig,
                                 fw_info: FrameworkInfo,
                                 fw_impl: FrameworkImplementation,
                                 tb_w: TensorboardWriter = None, ) -> Graph:
    """
     Add statistics moment correction on graph.
     Args:
        transformed_graph: Graph to add statistics correction.
        core_config (CoreConfig): Configuration object containing parameters of how the model should be
         quantized, including mixed precision parameters.
        fw_info: FrameworkInfo object with information about the specific framework's model.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tb_w (TensorboardWriter): TensorboardWriter object to use for logging events such as graphs, histograms, etc.

     Returns:
         Graph after statistics correction correction.
     """

    ######################################
    # Second Moment Correction
    ######################################
    tg_with_bias = substitute(transformed_graph, fw_impl.get_substitutions_statistics_correction(
        core_config.quantization_config))

    ########################################################
    # Compute bias correction to nodes' config candidates
    ########################################################
    tg_with_bias = compute_bias_correction_of_graph(tg_with_bias,
                                                    fw_info,
                                                    fw_impl)

    if tb_w is not None:
        tb_w.add_graph(tg_with_bias, 'statistics_computation')

    return tg_with_bias


def apply_statistics_correction(transformed_graph: Graph,
                                representative_data_gen: Callable,
                                core_config: CoreConfig,
                                fw_info: FrameworkInfo,
                                fw_impl: FrameworkImplementation,
                                tb_w: TensorboardWriter = None, ) -> Graph:
    """
     Apply statistics moment correction on graph.
     Args:
        transformed_graph: Graph to apply statistics correction.
        representative_data_gen (Callable): Dataset used for calibration.
        core_config (CoreConfig): Configuration object containing parameters of how the model should be
         quantized, including mixed precision parameters.
        fw_info: FrameworkInfo object with information about the specific framework's model.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tb_w (TensorboardWriter): TensorboardWriter object to use for logging events such as graphs, histograms, etc.

     Returns:
         Graph after statistics correction correction.
     """

    #############################################
    # Apply Second Moment Correction
    #############################################
    if core_config.quantization_config.weights_second_moment_correction:
        transformed_graph = apply_second_moment_correction_to_graph(transformed_graph, representative_data_gen,
                                                                    core_config, fw_info, fw_impl)

    #############################################
    # Apply Bias Correction
    #############################################
    if core_config.quantization_config.weights_bias_correction:
        transformed_graph = apply_bias_correction_to_graph(transformed_graph,
                                                           core_config,
                                                           fw_impl=fw_impl)
    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_statistics_correction')

    return transformed_graph
