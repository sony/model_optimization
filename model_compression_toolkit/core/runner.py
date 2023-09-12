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


import os
from typing import Callable, Tuple, Any, List, Dict

import numpy as np
from tqdm import tqdm

from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI, KPITarget
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_aggregation_methods import MpKpiAggregation
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_functions_mapping import kpi_functions_mapping
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_methods import MpKpiMetric
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_facade import search_bit_width
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.common.statistics_correction.statistics_correction import \
    statistics_correction_runner
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.core.common.visualization.final_config_visualizer import \
    WeightsFinalBitwidthConfigVisualizer, \
    ActivationFinalBitwidthConfigVisualizer
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter


def core_runner(in_model: Any,
                representative_data_gen: Callable,
                core_config: CoreConfig,
                fw_info: FrameworkInfo,
                fw_impl: FrameworkImplementation,
                tpc: TargetPlatformCapabilities,
                target_kpi: KPI = None,
                tb_w: TensorboardWriter = None):
    """
    Quantize a trained model using post-training quantization.
    First, the model graph is optimized using several transformations (e.g. folding BatchNormalization to preceding
    layers).
    Second, statistics (e.g. min/max, histogram, etc.) are collected for each layer's output
    (and input, depends on the quantization configuration) using a given representative dataset.
    Next, quantization parameters are calculated using the collected statistics
    (both coefficients and activations by default).

    Args:
        in_model: Model to quantize.
        representative_data_gen: Dataset used for calibration.
        core_config: CoreConfig containing parameters of how the model should be quantized
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.).
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tpc: TargetPlatformCapabilities object that models the inference target platform and
                                              the attached framework operator's information.
        target_kpi: KPI to constraint the search of the mixed-precision configuration for the model.
        tb_w: TensorboardWriter object for logging

    Returns:
        An internal graph representation of the input model.

    """

    graph = graph_preparation_runner(in_model,
                                     representative_data_gen,
                                     core_config.quantization_config,
                                     fw_info,
                                     fw_impl,
                                     tpc,
                                     tb_w,
                                     mixed_precision_enable=core_config.mixed_precision_enable)

    tg = _prepare_model_for_quantization(graph,
                                         representative_data_gen,
                                         core_config,
                                         fw_info,
                                         tb_w,
                                         fw_impl)

    ######################################
    # Finalize bit widths
    ######################################
    if target_kpi is not None:
        assert core_config.mixed_precision_enable
        if core_config.mixed_precision_config.configuration_overwrite is None:

            bit_widths_config = search_bit_width(tg,
                                                 fw_info,
                                                 fw_impl,
                                                 target_kpi,
                                                 core_config.mixed_precision_config,
                                                 representative_data_gen)
        else:
            Logger.warning(
                f'Mixed Precision has overwrite bit-width configuration{core_config.mixed_precision_config.configuration_overwrite}')
            bit_widths_config = core_config.mixed_precision_config.configuration_overwrite

    else:
        bit_widths_config = []

    tg = set_bit_widths(core_config.mixed_precision_enable,
                        tg,
                        bit_widths_config)

    # Edit the graph again after finalizing the configurations.
    # This is since some actions regard the final configuration and should be edited.
    edit_network_graph(tg, fw_info, core_config.debug_config.network_editor)

    _set_final_kpi(graph=tg,
                   final_bit_widths_config=bit_widths_config,
                   kpi_functions_dict=kpi_functions_mapping,
                   fw_info=fw_info,
                   fw_impl=fw_impl)

    if target_kpi is not None:
        # Retrieve lists of tuples (node, node's final weights/activation bitwidth)
        weights_conf_nodes_bitwidth = tg.get_final_weights_config()
        activation_conf_nodes_bitwidth = tg.get_final_activation_config()

        Logger.info(
            f'Final weights bit-width configuration: {[node_b[1] for node_b in weights_conf_nodes_bitwidth]}')
        Logger.info(
            f'Final activation bit-width configuration: {[node_b[1] for node_b in activation_conf_nodes_bitwidth]}')

        if tb_w is not None:
            if len(weights_conf_nodes_bitwidth) > 0:
                visual = WeightsFinalBitwidthConfigVisualizer(weights_conf_nodes_bitwidth)
                figure = visual.plot_config_bitwidth()
                tb_w.add_figure(figure, f'Weights final bit-width config')
            if len(activation_conf_nodes_bitwidth) > 0:
                visual = ActivationFinalBitwidthConfigVisualizer(activation_conf_nodes_bitwidth)
                figure = visual.plot_config_bitwidth()
                tb_w.add_figure(figure, f'Activation final bit-width config')

    return tg, bit_widths_config


def _init_tensorboard_writer(fw_info: FrameworkInfo) -> TensorboardWriter:
    """
    Create a TensorBoardWriter object initialized with the logger dir path if it was set,
    or None otherwise.

    Args:
        fw_info: FrameworkInfo object.

    Returns:
        A TensorBoardWriter object.
    """
    tb_w = None
    if Logger.LOG_PATH is not None:
        tb_log_dir = os.path.join(os.getcwd(), Logger.LOG_PATH, 'tensorboard_logs')
        Logger.info(f'To use Tensorboard, please run: tensorboard --logdir {tb_log_dir}')
        tb_w = TensorboardWriter(tb_log_dir, fw_info)
    return tb_w


def read_model_to_graph(in_model: Any,
                        representative_data_gen: Callable,
                        tpc: TargetPlatformCapabilities,
                        fw_info: FrameworkInfo = None,
                        fw_impl: FrameworkImplementation = None) -> Graph:

    """
    Read a model into a graph object.
    Args:
        in_model: Model to optimize and prepare for quantization.
        representative_data_gen: Dataset used for calibration.
        tpc: TargetPlatformCapabilities object that models the inference target platform and
                      the attached framework operator's information.
        fw_info: Information needed for quantization about the specific framework (e.g.,
                kernel channels indices, groups of layers by how they should be quantized, etc.)
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
    Returns:
        Graph object that represents the model.
    """
    graph = fw_impl.model_reader(in_model,
                                 representative_data_gen)
    graph.set_fw_info(fw_info)
    graph.set_tpc(tpc)
    return graph


def _prepare_model_for_quantization(transformed_graph: Graph,
                                    representative_data_gen: Callable,
                                    core_config: CoreConfig = CoreConfig(),
                                    fw_info: FrameworkInfo = None,
                                    tb_w: TensorboardWriter = None,
                                    fw_impl: FrameworkImplementation = None) -> Graph:
    """
    Prepare a trained model for post-training quantization.
    First, the model graph is optimized using several transformations (e.g. folding BatchNormalization to preceding layers).
    Second, statistics (e.g. min/max, histogram, etc.) are collected for each layer's output
    (and input, depends on the quantization configuration) using a given representative dataset.
    Next, quantization parameters are calculated using the collected statistics.
    Finally, more transformations (based on the statistics) are applied to increase the model's performance.

    Args:
        representative_data_gen (Callable): Dataset used for calibration.
        core_config (CoreConfig): CoreConfig containing parameters of how the model should be quantized.
        fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g.,
        kernel channels indices, groups of layers by how they should be quantized, etc.)
        tb_w (TensorboardWriter): TensorboardWriter object to use for logging events such as graphs, histograms, etc.
        fw_impl (FrameworkImplementation): FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Graph object that represents the model, contains thresholds, and ready for quantization.
    """

    ######################################
    # Graph analyzing (attaching statistics collectors)
    ######################################
    analyzer_graph(fw_impl.attach_sc_to_node,
                   transformed_graph,
                   fw_info,
                   core_config.quantization_config)  # Mark points for statistics collection

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_analyzer_graph')

    ######################################
    # Statistic collection
    ######################################
    mi = ModelCollector(transformed_graph,
                        fw_impl,
                        fw_info)

    for _data in tqdm(representative_data_gen()):
        mi.infer(_data)

    ######################################
    # Edit network according to user
    # specific settings
    ######################################
    # Notice that not all actions affect at this stage (for example, actions that edit the final configuration as
    # there are no final configurations at this stage of the optimization). For this reason we edit the graph
    # again at the end of the optimization process.
    edit_network_graph(transformed_graph, fw_info, core_config.debug_config.network_editor)

    ######################################
    # Calculate quantization params
    ######################################
    calculate_quantization_params(transformed_graph,
                                  fw_info,
                                  fw_impl=fw_impl)

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'thresholds_selection')
        tb_w.add_all_statistics(transformed_graph, 'thresholds_selection')

    ######################################
    # Graph substitution (post statistics collection)
    ######################################
    transformed_graph = substitute(transformed_graph,
                                   fw_impl.get_substitutions_post_statistics_collection(core_config.quantization_config))

    ######################################
    # Shift Negative Activations
    ######################################
    if core_config.quantization_config.shift_negative_activation_correction:
        transformed_graph = fw_impl.shift_negative_correction(transformed_graph,
                                                              core_config,
                                                              fw_info)
        if tb_w is not None:
            tb_w.add_graph(transformed_graph, 'after_shift_negative_correction')
            tb_w.add_all_statistics(transformed_graph, 'after_shift_negative_correction')

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'post_statistics_collection_substitutions')
        tb_w.add_all_statistics(transformed_graph, 'post_statistics_collection_substitutions')

    ######################################
    # Statistics Correction
    ######################################
    tg_with_bias = statistics_correction_runner(transformed_graph, core_config, fw_info, fw_impl, tb_w)

    for n in tg_with_bias.nodes:
        assert n.final_weights_quantization_cfg is None

    return tg_with_bias


def _set_final_kpi(graph: Graph,
                   final_bit_widths_config: List[int],
                   kpi_functions_dict: Dict[KPITarget, Tuple[MpKpiMetric, MpKpiAggregation]],
                   fw_info: FrameworkInfo,
                   fw_impl: FrameworkImplementation):
    """
    Computing the KPIs of the model according to the final bit-width configuration,
    and setting it (inplace) in the graph's UserInfo field.

    Args:
        graph: Graph to compute the KPI for.
        final_bit_widths_config: The final bit-width configuration to quantize the model accordingly.
        kpi_functions_dict: A mapping between a KPITarget and a pair of kpi method and kpi aggregation functions.
        fw_info: A FrameworkInfo object.
        fw_impl: FrameworkImplementation object with specific framework methods implementation.

    """

    final_kpis_dict = {}
    for kpi_target, kpi_funcs in kpi_functions_dict.items():
        kpi_method, kpi_aggr = kpi_funcs
        if kpi_target == KPITarget.BOPS:
            final_kpis_dict[kpi_target] = kpi_aggr(kpi_method(final_bit_widths_config, graph, fw_info, fw_impl, False), False)[0]
        else:
            non_conf_kpi = kpi_method([], graph, fw_info, fw_impl)
            conf_kpi = kpi_method(final_bit_widths_config, graph, fw_info, fw_impl)
            if len(final_bit_widths_config) > 0 and len(non_conf_kpi) > 0:
                final_kpis_dict[kpi_target] = kpi_aggr(np.concatenate([conf_kpi, non_conf_kpi]), False)[0]
            elif len(final_bit_widths_config) > 0 and len(non_conf_kpi) == 0:
                final_kpis_dict[kpi_target] = kpi_aggr(conf_kpi, False)[0]
            elif len(final_bit_widths_config) == 0 and len(non_conf_kpi) > 0:
                # final_bit_widths_config == 0 ==> no configurable nodes,
                # thus, KPI can be computed from non_conf_kpi alone
                final_kpis_dict[kpi_target] = kpi_aggr(non_conf_kpi, False)[0]
            else:
                # No relevant nodes have been quantized with affect on the given target - since we only consider
                # in the model's final size the quantized layers size, this means that the final size for this target
                # is zero.
                Logger.warning(f"No relevant quantized layers for the KPI target {kpi_target} were found, the recorded"
                               f"final KPI for this target would be 0.")
                final_kpis_dict[kpi_target] = 0

    final_kpi = KPI()
    final_kpi.set_kpi_by_target(final_kpis_dict)
    graph.user_info.final_kpi = final_kpi
    graph.user_info.mixed_precision_cfg = final_bit_widths_config
