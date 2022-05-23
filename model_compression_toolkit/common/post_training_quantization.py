# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
import os
from functools import partial
from typing import Callable, List, Tuple, Any
from tqdm import tqdm

from model_compression_toolkit import common
from model_compression_toolkit.common import Logger
from model_compression_toolkit.common.gptq.gptq_config import GradientPTQConfig
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common import FrameworkInfo
from model_compression_toolkit.common.constants import NUM_SAMPLES_CS_TENSORBOARD
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.common.gptq.gptq_training import gptq_training
from model_compression_toolkit.common.mixed_precision.mixed_precision_search_facade import search_bit_width
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.common.quantization.filter_nodes_candidates import filter_nodes_candidates
from model_compression_toolkit.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.common.bias_correction.compute_bias_correction_of_graph import \
    compute_bias_correction_of_graph

from model_compression_toolkit.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.common.quantization.quantization_config import DEFAULTCONFIG
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.core_config import CoreConfig
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params

from model_compression_toolkit.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph

from model_compression_toolkit.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.common.substitutions.linear_collapsing_substitution import linear_collapsing_substitute
from model_compression_toolkit.common.user_info import UserInformation
from model_compression_toolkit.common.model_collector import ModelCollector

from model_compression_toolkit.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.common.bias_correction.apply_bias_correction_to_graph import \
    apply_bias_correction_to_graph
from model_compression_toolkit.common.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.common.visualization.final_config_visualizer import WeightsFinalBitwidthConfigVisualizer, \
    ActivationFinalBitwidthConfigVisualizer


def post_training_quantization(in_model: Any,
                               representative_data_gen: Callable,
                               core_config: CoreConfig,
                               fw_info: FrameworkInfo,
                               fw_impl: FrameworkImplementation,
                               tpc: TargetPlatformCapabilities,
                               gptq_config: GradientPTQConfig = None,
                               target_kpi: KPI = None):
    """
    Quantize a trained model using post-training quantization.
    First, the model graph is optimized using several transformations (e.g. folding BatchNormalization to preceding
    layers).
    Second, statistics (e.g. min/max, histogram, etc.) are collected for each layer's output
    (and input, depends on the quantization configuration) using a given representative dataset.
    Next, quantization parameters are calculated using the collected statistics and the model is quantized
    (both coefficients and activations by default).
    If a gptq configuration is passed, the quantized weights are optimized using knowledge
    distillation. This is done by comparing points between the float and quantized models, and minimizing the
    observed loss.

    Args:
        in_model: Model to quantize.
        representative_data_gen: Dataset used for calibration.
        core_config: CoreConfig containing parameters of how the model should be quantized. `Default
        configuration. <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10
        /model_compression_toolkit/common/quantization/quantization_config.py#L163>`_
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.). `Default Keras info
        <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10
        /model_compression_toolkit/keras/default_framework_info.py#L114>`_
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        tpc: TargetPlatformCapabilities object that models the inference target platform and
                                              the attached framework operator's information.
        gptq_config: Configuration for using gradient-based PTQ (e.g. optimizer).
        target_kpi: KPI to constraint the search of the mixed-precision configuration for the model.

    Returns:
        A quantized model and information the user may need to handle the quantized model.

    """

    tb_w = _init_tensorboard_writer(fw_info)

    graph = read_model_to_graph(in_model,
                                representative_data_gen,
                                tpc,
                                fw_info,
                                fw_impl)

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
                                                 core_config.mixed_precision_config,
                                                 fw_info,
                                                 target_kpi,
                                                 partial(fw_impl.get_sensitivity_evaluation_fn,
                                                         representative_data_gen=representative_data_gen,
                                                         fw_info=fw_info))
        else:
            Logger.warning(
                f'Mixed Precision has overwrite bit-width configuration{core_config.mixed_precision_config.configuration_overwrite}')
            bit_widths_config = core_config.mixed_precision_config.configuration_overwrite

    else:
        bit_widths_config = None

    tg = set_bit_widths(core_config.mixed_precision_enable,
                        tg,
                        fw_info,
                        bit_widths_config)

    # Retrive lists of tuples (node, node's final weights/activation bitwidth)
    weights_conf_nodes_bitwidth = tg.get_final_weights_config()
    activation_conf_nodes_bitwidth = tg.get_final_activation_config()

    common.Logger.info(f'Approximated model size (in bytes): {tg.get_memory()}')
    common.Logger.info(f'Approximated compression ratio: {round(graph.get_float_memory() / (tg.get_memory() + 1e-8), 3)}')
    common.Logger.info(
        f'Final weights bit-width configuration: {[node_b[1] for node_b in weights_conf_nodes_bitwidth]}')
    common.Logger.info(
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

    quantized_model, user_info = _quantize_fixed_bit_widths_graph(core_config.debug_config.analyze_similarity,
                                                                  fw_info,
                                                                  gptq_config,
                                                                  representative_data_gen,
                                                                  tb_w,
                                                                  tg,
                                                                  fw_impl)
    user_info.mixed_precision_cfg = bit_widths_config

    return quantized_model, user_info


def get_finalized_graph(initial_graph: Graph,
                        quant_config: QuantizationConfig = DEFAULTCONFIG,
                        fw_info: FrameworkInfo = None,
                        tb_w: TensorboardWriter = None,
                        fw_impl: FrameworkImplementation = None,
                        mixed_precision_enable: bool = False) -> Graph:
    """
    Applies all edit operation (edit, substitutions, etc.) on the model's graph, to prepare it for the quantization
    process. All future graph substitutions and operations that change the graph should be added to this method.

    Args:
        initial_graph (Graph): Graph to apply the changes to.
        quant_config (QuantizationConfig): QuantizationConfig containing parameters of how the model should be
        quantized.
        fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g.,
        kernel channels indices, groups of layers by how they should be quantized, etc.)
        tb_w (TensorboardWriter): TensorboardWriter object to use for logging events such as graphs, histograms, etc.
        fw_impl (FrameworkImplementation): FrameworkImplementation object with a specific framework methods implementation.
        mixed_precision_enable: is mixed precision enabled.

    Returns: Graph object that represents the model, after applying all required modifications to it.

    """

    ######################################
    # Graph substitution (prepare graph)
    ######################################
    graph = substitute(initial_graph, fw_impl.get_substitutions_prepare_graph())

    if tb_w is not None:
        tb_w.add_graph(graph, 'after_graph_preparation')

    #########################################
    # Set prior info to nodes
    ##########################################
    for node in graph.nodes:
        node.prior_info = fw_impl.get_node_prior_info(node=node,
                                                      fw_info=fw_info,
                                                      graph=graph)
    ##################################################
    # Graph substitution (pre statistics collection)
    ##################################################
    transformed_graph = substitute(graph, fw_impl.get_substitutions_pre_statistics_collection(quant_config))
    if quant_config.linear_collapsing:
        transformed_graph = linear_collapsing_substitute(transformed_graph, fw_impl.get_linear_collapsing_substitution())
    if quant_config.residual_collapsing:
        transformed_graph = substitute(transformed_graph, fw_impl.get_residual_collapsing_substitution())

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'pre_statistics_collection_substitutions')

    ######################################
    # Add quantization configurations
    ######################################
    transformed_graph = set_quantization_configuration_to_graph(graph=transformed_graph,
                                                                quant_config=quant_config,
                                                                mixed_precision_enable=mixed_precision_enable)

    ######################################
    # Graph marking points
    ######################################
    transformed_graph = substitute(transformed_graph, fw_impl.get_substitutions_marking())

    ######################################
    # Channel equalization
    ######################################
    transformed_graph = substitute(transformed_graph,
                                   fw_impl.get_substitutions_channel_equalization(quant_config,
                                                                                  fw_info))

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_graph_marking')

    ######################################
    # Filter nodes' candidates
    ######################################
    transformed_graph = filter_nodes_candidates(transformed_graph)

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_candidates_filtering')

    return transformed_graph


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
    if common.Logger.LOG_PATH is not None:
        tb_log_dir = os.path.join(os.getcwd(), common.Logger.LOG_PATH, 'tensorboard_logs')
        common.Logger.info(f'To use Tensorboard, please run: tensorboard --logdir {tb_log_dir}')
        tb_w = TensorboardWriter(tb_log_dir, fw_info)
    return tb_w


def _quantize_model(fw_info: FrameworkInfo,
                    tb_w: TensorboardWriter,
                    tg: Graph,
                    fw_impl: FrameworkImplementation) -> Tuple[Any, UserInformation]:
    """
    Quantize graph's weights, and build a quantized framework model from it.

    Args:
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).
        tb_w: TensorBoardWriter object to log events.
        tg: A prepared for quantization graph.

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


def _analyze_similarity(representative_data_gen: Callable,
                        tb_w: TensorboardWriter,
                        tg: Graph,
                        tg_float: Graph):
    """
    Plot the cosine similarity of different points on the graph between the float and quantized
    graphs. Add them to the passed TensorboardWriter object and close all tensorboard writer open
    files.

    Args:
        representative_data_gen: Dataset used for calibration.
        tb_w: TensorBoardWriter object to log events.
        tg: Graph of quantized model.
        tg_float: Graph of float model.

    """
    if tb_w is not None:
        visual = KerasNNVisualizer(tg_float, tg)
        for i in range(NUM_SAMPLES_CS_TENSORBOARD):
            figure = visual.plot_cs_graph(representative_data_gen())
            tb_w.add_figure(figure, f'cosine_similarity_sample_{i}')
        tb_w.close()


def _apply_gptq(gptq_config: GradientPTQConfig,
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
    if gptq_config is not None:
        common.Logger.info("Using experimental Gradient Based PTQ: If you encounter an issue "
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


def _quantize_fixed_bit_widths_graph(analyze_similarity: bool,
                                     fw_info: FrameworkInfo,
                                     gptq_config: GradientPTQConfig,
                                     representative_data_gen: Callable,
                                     tb_w: TensorboardWriter,
                                     tg: Graph,
                                     fw_impl: FrameworkImplementation) -> Tuple[Any, UserInformation]:
    """
    Quantize a graph that has final weights candidates quantization configurations.
    Before we quantize the graph weights, we apply GPTQ to get an improved graph.

    Args:
        analyze_similarity: Whether to plot similarity figures within TensorBoard (when logger is enabled) or not.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.)
        gptq_config: Configuration for using GPTQ (e.g. optimizer).
        representative_data_gen: Dataset used for GPTQ fine tuning.
        tb_w: A TensorBoardWriter object initialized with the logger dir path if it was set, or None otherwise.
        tg: Graph to apply GPTQ and to quantize.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        A tuple of the quantized model and an object of UserInformation.

    """
    #############################################
    # Apply Bias Correction
    #############################################
    tg_bias = apply_bias_correction_to_graph(tg,
                                             fw_info=fw_info,
                                             fw_impl=fw_impl)
    if tb_w is not None:
        tb_w.add_graph(tg_bias, 'after_bias_correction')
    #############################################
    # Gradient Based Post Training Quantization
    #############################################
    tg_bias = _apply_gptq(gptq_config,
                          representative_data_gen,
                          tb_w,
                          tg,
                          tg_bias,
                          fw_info,
                          fw_impl)

    tg_float = copy.deepcopy(tg)  # Copy graph before quantization (for similarity analyzer)
    ######################################
    # Final Model Quantization
    ######################################
    quantized_model, user_info = _quantize_model(fw_info,
                                                 tb_w,
                                                 tg_bias,
                                                 fw_impl)
    if analyze_similarity:
        _analyze_similarity(representative_data_gen,
                            tb_w,
                            tg_bias,
                            tg_float)

    return quantized_model, user_info


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


def _prepare_model_for_quantization(graph: Graph,
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
        change quantization settings of the filtered nodes.
        core_config (CoreConfig): CoreConfig containing parameters of how the model should be
        quantized.
        fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g.,
        kernel channels indices, groups of layers by how they should be quantized, etc.)
        tb_w (TensorboardWriter): TensorboardWriter object to use for logging events such as graphs, histograms, etc.
        fw_impl (FrameworkImplementation): FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Graph object that represents the model, contains thresholds, and ready for quantization.
    """
    if tb_w is not None:
        tb_w.add_graph(graph, 'initial_graph')

    transformed_graph = get_finalized_graph(graph,
                                            core_config.quantization_config,
                                            fw_info,
                                            tb_w,
                                            fw_impl,
                                            mixed_precision_enable=core_config.mixed_precision_enable)

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

    for _ in tqdm(range(core_config.n_iter)):
        mi.infer(representative_data_gen())

    ######################################
    # Edit network according to user specific settings
    ######################################
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

    ########################################################
    # Compute bias correction to nodes' config candidates
    ########################################################
    tg_with_bias = compute_bias_correction_of_graph(transformed_graph,
                                                    fw_info,
                                                    fw_impl)

    if tb_w is not None:
        tb_w.add_graph(tg_with_bias, 'bias_correction_computation')

    for n in tg_with_bias.nodes:
        assert n.final_weights_quantization_cfg is None

    return tg_with_bias
