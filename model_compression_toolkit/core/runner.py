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


from typing import Callable, Tuple, Any, List, Dict

import numpy as np

from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.hessian.hessian_info_service import HessianInfoService
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.quantization_prep_runner import quantization_preparation_runner
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI, KPITarget
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_aggregation_methods import MpKpiAggregation
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_functions_mapping import kpi_functions_mapping
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_methods import MpKpiMetric
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_facade import search_bit_width
from model_compression_toolkit.core.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.core.common.visualization.final_config_visualizer import \
    WeightsFinalBitwidthConfigVisualizer, \
    ActivationFinalBitwidthConfigVisualizer
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter, \
    finalize_bitwidth_in_tb


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

    # Warn is representative dataset has batch-size == 1
    batch_data = iter(representative_data_gen()).__next__()
    if isinstance(batch_data, list):
        batch_data = batch_data[0]
    if batch_data.shape[0] == 1:
        Logger.warning('representative_data_gen generates a batch size of 1 which can be slow for optimization:'
                       ' consider increasing the batch size')

    graph = graph_preparation_runner(in_model,
                                     representative_data_gen,
                                     core_config.quantization_config,
                                     fw_info,
                                     fw_impl,
                                     tpc,
                                     tb_w,
                                     mixed_precision_enable=core_config.mixed_precision_enable)

    hessian_info_service = HessianInfoService(graph=graph,
                                              representative_dataset=representative_data_gen,
                                              fw_impl=fw_impl)

    tg = quantization_preparation_runner(graph=graph,
                                         representative_data_gen=representative_data_gen,
                                         core_config=core_config,
                                         fw_info=fw_info,
                                         fw_impl=fw_impl,
                                         tb_w=tb_w)

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
                                                 representative_data_gen,
                                                 hessian_info_service=hessian_info_service)
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
            finalize_bitwidth_in_tb(tb_w, weights_conf_nodes_bitwidth, activation_conf_nodes_bitwidth)

    return tg, bit_widths_config, hessian_info_service


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
