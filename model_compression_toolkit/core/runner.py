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
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_data import \
    requires_mixed_precision
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.quantization_prep_runner import quantization_preparation_runner
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization, RUTarget
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.ru_aggregation_methods import MpRuAggregation
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.ru_functions_mapping import ru_functions_mapping
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.ru_methods import MpRuMetric
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
                target_resource_utilization: ResourceUtilization = None,
                running_gptq: bool = False,
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
        target_resource_utilization: ResourceUtilization to constraint the search of the mixed-precision configuration for the model.
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

    # Checking whether to run mixed precision quantization
    if target_resource_utilization is not None:
        if core_config.mixed_precision_config is None:
            Logger.critical("Provided an initialized target_resource_utilization, that means that mixed precision quantization is "
                            "enabled, but the provided MixedPrecisionQuantizationConfig is None.")
        # Determine whether to use mixed precision or single precision based on target_resource_utilization.
        if requires_mixed_precision(in_model,
                                    target_resource_utilization,
                                    representative_data_gen,
                                    core_config,
                                    tpc,
                                    fw_info,
                                    fw_impl):
            core_config.mixed_precision_config.set_mixed_precision_enable()
            Logger.info('Mixed precision enabled.')

    graph = graph_preparation_runner(in_model,
                                     representative_data_gen,
                                     core_config.quantization_config,
                                     fw_info,
                                     fw_impl,
                                     tpc,
                                     tb_w,
                                     mixed_precision_enable=core_config.mixed_precision_enable,
                                     running_gptq=running_gptq)

    hessian_info_service = HessianInfoService(graph=graph, representative_dataset_gen=representative_data_gen,
                                              fw_impl=fw_impl)

    tg = quantization_preparation_runner(graph=graph,
                                         representative_data_gen=representative_data_gen,
                                         core_config=core_config,
                                         fw_info=fw_info,
                                         fw_impl=fw_impl,
                                         tb_w=tb_w,
                                         hessian_info_service=hessian_info_service)

    ######################################
    # Finalize bit widths
    ######################################
    if core_config.mixed_precision_enable:
        if core_config.mixed_precision_config.configuration_overwrite is None:

            bit_widths_config = search_bit_width(tg,
                                                 fw_info,
                                                 fw_impl,
                                                 target_resource_utilization,
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

    _set_final_resource_utilization(graph=tg,
                                    final_bit_widths_config=bit_widths_config,
                                    ru_functions_dict=ru_functions_mapping,
                                    fw_info=fw_info,
                                    fw_impl=fw_impl)

    if core_config.mixed_precision_enable:
        # Retrieve lists of tuples (node, node's final weights/activation bitwidth)
        weights_conf_nodes_bitwidth = tg.get_final_weights_config(fw_info)
        activation_conf_nodes_bitwidth = tg.get_final_activation_config()

        if len(weights_conf_nodes_bitwidth) > 0:
            Logger.info(
                f'Final weights bit-width configuration: {[node_b[1] for node_b in weights_conf_nodes_bitwidth]}')

        if len(activation_conf_nodes_bitwidth) > 0:
            Logger.info(
                f'Final activation bit-width configuration: {[node_b[1] for node_b in activation_conf_nodes_bitwidth]}')

        if tb_w is not None:
            finalize_bitwidth_in_tb(tb_w, weights_conf_nodes_bitwidth, activation_conf_nodes_bitwidth)

    return tg, bit_widths_config, hessian_info_service


def _set_final_resource_utilization(graph: Graph,
                                    final_bit_widths_config: List[int],
                                    ru_functions_dict: Dict[RUTarget, Tuple[MpRuMetric, MpRuAggregation]],
                                    fw_info: FrameworkInfo,
                                    fw_impl: FrameworkImplementation):
    """
    Computing the resource utilization of the model according to the final bit-width configuration,
    and setting it (inplace) in the graph's UserInfo field.

    Args:
        graph: Graph to compute the resource utilization for.
        final_bit_widths_config: The final bit-width configuration to quantize the model accordingly.
        ru_functions_dict: A mapping between a RUTarget and a pair of resource utilization method and resource utilization aggregation functions.
        fw_info: A FrameworkInfo object.
        fw_impl: FrameworkImplementation object with specific framework methods implementation.

    """

    final_ru_dict = {}
    for ru_target, ru_funcs in ru_functions_dict.items():
        ru_method, ru_aggr = ru_funcs
        if ru_target == RUTarget.BOPS:
            final_ru_dict[ru_target] = \
            ru_aggr(ru_method(final_bit_widths_config, graph, fw_info, fw_impl, False), False)[0]
        else:
            non_conf_ru = ru_method([], graph, fw_info, fw_impl)
            conf_ru = ru_method(final_bit_widths_config, graph, fw_info, fw_impl)
            if len(final_bit_widths_config) > 0 and len(non_conf_ru) > 0:
                final_ru_dict[ru_target] = ru_aggr(np.concatenate([conf_ru, non_conf_ru]), False)[0]
            elif len(final_bit_widths_config) > 0 and len(non_conf_ru) == 0:
                final_ru_dict[ru_target] = ru_aggr(conf_ru, False)[0]
            elif len(final_bit_widths_config) == 0 and len(non_conf_ru) > 0:
                # final_bit_widths_config == 0 ==> no configurable nodes,
                # thus, ru can be computed from non_conf_ru alone
                final_ru_dict[ru_target] = ru_aggr(non_conf_ru, False)[0]
            else:
                # No relevant nodes have been quantized with affect on the given target - since we only consider
                # in the model's final size the quantized layers size, this means that the final size for this target
                # is zero.
                Logger.warning(f"No relevant quantized layers for the ru target {ru_target} were found, the recorded "
                               f"final ru for this target would be 0.")
                final_ru_dict[ru_target] = 0

    final_ru = ResourceUtilization()
    final_ru.set_resource_utilization_by_target(final_ru_dict)
    graph.user_info.final_resource_utilization = final_ru
    graph.user_info.mixed_precision_cfg = final_bit_widths_config
