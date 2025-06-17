# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

from tqdm import tqdm

from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.hessian import HessianInfoService
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.common.statistics_correction.statistics_correction import \
    statistics_correction_runner
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute

from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter


def quantization_preparation_runner(graph: Graph,
                                    representative_data_gen: Callable,
                                    core_config: CoreConfig,
                                    fw_impl: FrameworkImplementation,
                                    tb_w: TensorboardWriter = None,
                                    hessian_info_service: HessianInfoService = None, ) -> Graph:
    """
    Prepares a trained model for post-training quantization.
    First, the model graph is optimized using several transformations (e.g. folding BatchNormalization to preceding layers).
    Second, statistics (e.g. min/max, histogram, etc.) are collected for each layer's output
    (and input, depends on the quantization configuration) using a given representative dataset.
    Next, quantization parameters are calculated using the collected statistics.
    Finally, more transformations (based on the statistics) are applied to increase the model's performance.

    Args:
        graph: A graph representation of the model to be quantized.
        representative_data_gen: Dataset used for calibration.
        core_config: CoreConfig containing parameters of how the model should be quantized
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tb_w: TensorboardWriter object for logging
        hessian_info_service: HessianInfoService object for retrieving Hessian-based scores.

    Returns:
        Graph object that represents the model, contains thresholds, and ready for quantization.
    """

    ######################################
    # Statistic collection
    ######################################
    mi = ModelCollector(graph,
                        fw_impl,
                        hessian_info_service,
                        core_config.quantization_config)  # Mark points for statistics collection

    for _data in tqdm(representative_data_gen(), "Statistics Collection"):
        mi.infer(_data)

    if tb_w is not None:
        tb_w.add_graph(graph, 'after_statistic_collection')

    ######################################
    # Edit network according to user
    # specific settings
    ######################################
    # Notice that not all actions affect at this stage (for example, actions that edit the final configuration as
    # there are no final configurations at this stage of the optimization). For this reason we edit the graph
    # again at the end of the optimization process.
    edit_network_graph(graph, core_config.debug_config.network_editor)

    ######################################
    # Calculate quantization params
    ######################################

    calculate_quantization_params(graph, fw_impl=fw_impl, repr_data_gen_fn=representative_data_gen,
                                  hessian_info_service=hessian_info_service)

    if tb_w is not None:
        tb_w.add_graph(graph, 'thresholds_selection')
        tb_w.add_all_statistics(graph, 'thresholds_selection')

    ######################################
    # Graph substitution (post statistics collection)
    ######################################
    transformed_graph = substitute(graph,
                                   fw_impl.get_substitutions_post_statistics_collection(core_config.quantization_config))

    ######################################
    # Shift Negative Activations
    ######################################
    if core_config.quantization_config.shift_negative_activation_correction:
        transformed_graph = fw_impl.shift_negative_correction(transformed_graph,
                                                              core_config)
        if tb_w is not None:
            tb_w.add_graph(transformed_graph, 'after_shift_negative_correction')
            tb_w.add_all_statistics(transformed_graph, 'after_shift_negative_correction')

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'post_statistics_collection_substitutions')
        tb_w.add_all_statistics(transformed_graph, 'post_statistics_collection_substitutions')

    ######################################
    # Statistics Correction
    ######################################
    tg_with_bias = statistics_correction_runner(transformed_graph, core_config, fw_impl, tb_w)

    for n in tg_with_bias.nodes:
        assert n.final_weights_quantization_cfg is None

    return tg_with_bias
