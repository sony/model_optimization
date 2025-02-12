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

import copy
from typing import Callable, Any, List, Optional

from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.fusion.graph_fuser import GraphFuser
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.memory_graph.compute_graph_max_cut import compute_graph_max_cut, \
    SchedulerInfo
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import MemoryGraph
from model_compression_toolkit.core.common.hessian.hessian_info_service import HessianInfoService
from model_compression_toolkit.core.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_candidates_filter import \
    filter_candidates_for_mixed_precision
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_facade import search_bit_width
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    ResourceUtilizationCalculator, TargetInclusionCriterion, BitwidthMode
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_data import \
    requires_mixed_precision
from model_compression_toolkit.core.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter, \
    finalize_bitwidth_in_tb
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.quantization_prep_runner import quantization_preparation_runner
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities


def core_runner(in_model: Any,
                representative_data_gen: Callable,
                core_config: CoreConfig,
                fw_info: FrameworkInfo,
                fw_impl: FrameworkImplementation,
                fqc: FrameworkQuantizationCapabilities,
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
        fqc: FrameworkQuantizationCapabilities object that models the inference target platform and
                                              the attached framework operator's information.
        target_resource_utilization: ResourceUtilization to constraint the search of the mixed-precision configuration for the model.
        tb_w: TensorboardWriter object for logging

    Returns:
        An internal graph representation of the input model.

    """

    # Warn is representative dataset has batch-size == 1
    batch_data = next(iter(representative_data_gen()))
    if isinstance(batch_data, list):
        batch_data = batch_data[0]
    if batch_data.shape[0] == 1:
        Logger.warning('representative_data_gen generates a batch size of 1 which can be slow for optimization:'
                       ' consider increasing the batch size')

    # Checking whether to run mixed precision quantization
    if target_resource_utilization is not None and target_resource_utilization.is_any_restricted():
        if core_config.mixed_precision_config is None:  # pragma: no cover
            Logger.critical("Provided an initialized target_resource_utilization, that means that mixed precision quantization is "
                            "enabled, but the provided MixedPrecisionQuantizationConfig is None.")
        if target_resource_utilization.activation_restricted() or target_resource_utilization.total_mem_restricted():
            Logger.warning("Using an experimental feature max-cut for activation memory utilization estimation.")
        # Determine whether to use mixed precision or single precision based on target_resource_utilization.
        if requires_mixed_precision(in_model,
                                    target_resource_utilization,
                                    representative_data_gen,
                                    core_config,
                                    fqc,
                                    fw_info,
                                    fw_impl):
            core_config.mixed_precision_config.set_mixed_precision_enable()
            Logger.info('Mixed precision enabled.')

    graph = graph_preparation_runner(in_model,
                                     representative_data_gen,
                                     core_config.quantization_config,
                                     fw_info,
                                     fw_impl,
                                     fqc,
                                     core_config.bit_width_config,
                                     tb_w,
                                     mixed_precision_enable=core_config.is_mixed_precision_enabled,
                                     running_gptq=running_gptq)

    hessian_info_service = HessianInfoService(graph=graph, fw_impl=fw_impl)

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
    if core_config.is_mixed_precision_enabled:
        if core_config.mixed_precision_config.configuration_overwrite is None:

            filter_candidates_for_mixed_precision(graph, target_resource_utilization, fw_info, fqc)
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

        if target_resource_utilization.activation_restricted() or target_resource_utilization.total_mem_restricted():
            Logger.warning(
                f"Running mixed precision for activation compression, please note this feature is experimental and is "
                f"subject to future changes. If you encounter an issue, please open an issue in our GitHub "
                f"project https://github.com/sony/model_optimization")
    else:
        bit_widths_config = []

    tg = set_bit_widths(core_config.is_mixed_precision_enabled,
                        tg,
                        bit_widths_config)

    ######################################
    # Compute Activation Bias Correction
    ######################################
    if core_config.quantization_config.activation_bias_correction:
        tg = fw_impl.compute_activation_bias_correction(graph=tg,
                                                        quant_config=core_config.quantization_config,
                                                        fw_info=fw_info)

    # Edit the graph again after finalizing the configurations.
    # This is since some actions regard the final configuration and should be edited.
    edit_network_graph(tg, fw_info, core_config.debug_config.network_editor)

    _set_final_resource_utilization(graph=tg,
                                    final_bit_widths_config=bit_widths_config,
                                    target_resource_utilization=target_resource_utilization,
                                    fw_info=fw_info,
                                    fw_impl=fw_impl)

    if core_config.is_mixed_precision_enabled:
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

    scheduler_info = None
    if core_config.debug_config.simulate_scheduler:
        graph_to_fuse = copy.deepcopy(tg)
        fused_nodes_mapping = GraphFuser().create_fused_graph(graph_to_fuse)
        memory_graph = MemoryGraph(graph_to_fuse)
        schedule, max_cut, cuts = compute_graph_max_cut(memory_graph)
        scheduler_info = SchedulerInfo(
            operators_scheduling=schedule,
            max_cut=float(max_cut),
            cuts=cuts,
            fused_nodes_mapping=fused_nodes_mapping
        )

    return tg, bit_widths_config, hessian_info_service, scheduler_info


def _set_final_resource_utilization(graph: Graph,
                                    final_bit_widths_config: List[int],
                                    target_resource_utilization: Optional[ResourceUtilization],
                                    fw_info: FrameworkInfo,
                                    fw_impl: FrameworkImplementation):
    """
    Computing the resource utilization of the model according to the final bit-width configuration,
    and setting it (inplace) in the graph's UserInfo field.

    Args:
        graph: Graph to compute the resource utilization for.
        final_bit_widths_config: The final bit-width configuration to quantize the model accordingly.
        target_resource_utilization: Requested target resource utilization if relevant.
        fw_info: A FrameworkInfo object.
        fw_impl: FrameworkImplementation object with specific framework methods implementation.

    """
    ru_targets = target_resource_utilization.get_restricted_targets() if target_resource_utilization else None
    final_ru = None
    if ru_targets:
        ru_calculator = ResourceUtilizationCalculator(graph, fw_impl, fw_info)
        w_qcs = {n.name: n.final_weights_quantization_cfg for n in graph.nodes}
        a_qcs = {n.name: n.final_activation_quantization_cfg for n in graph.nodes}
        final_ru = ru_calculator.compute_resource_utilization(TargetInclusionCriterion.AnyQuantized,
                                                              BitwidthMode.QCustom, act_qcs=a_qcs, w_qcs=w_qcs,
                                                              ru_targets=ru_targets, allow_unused_qcs=True)
        summary = final_ru.get_summary_str(restricted=True)
        Logger.info(f'Resource utilization for quantized mixed-precision targets:\n {summary}.')
    graph.user_info.final_resource_utilization = final_ru
    graph.user_info.mixed_precision_cfg = final_bit_widths_config
