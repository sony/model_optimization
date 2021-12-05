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
from model_compression_toolkit.common.gptq.gptq_config import GradientPTQConfig
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common import FrameworkInfo
from model_compression_toolkit.common.constants import NUM_SAMPLES_CS_TENSORBOARD
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.mixed_precision.bit_width_setter import set_bit_widths

from model_compression_toolkit.common.mixed_precision.mixed_precision_search_facade import search_bit_width
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.common.network_editors.actions import EditRule
from model_compression_toolkit.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.common.bias_correction.compute_bias_correction_of_graph import compute_bias_correction_of_graph

from model_compression_toolkit.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.common.quantization.quantization_config import DEFAULTCONFIG
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params

from model_compression_toolkit.common.quantization.set_node_quantization_config import set_quantization_configuration_to_graph

from model_compression_toolkit.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.common.user_info import UserInformation
from model_compression_toolkit.common.model_collector import ModelCollector

from model_compression_toolkit.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.common.bias_correction.apply_bias_correction_to_graph import apply_bias_correction_to_graph





def post_training_quantization(in_model: Any,
                               representative_data_gen: Callable,
                               n_iter: int,
                               quant_config: QuantizationConfig,
                               fw_info: FrameworkInfo,
                               fw_impl: FrameworkImplementation,
                               network_editor: List[EditRule] = [],
                               gptq_config: GradientPTQConfig = None,
                               analyze_similarity: bool = False,
                               target_kpi: KPI = None):
    """
    Quantize a trained model using post-training quantization. The model is quantized using a
    symmetric constraint quantization thresholds (power of two).
    The model is first optimized using several transformations (e.g. BatchNormalization folding to
    preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
    being collected for each layer's output (and input, depends on the quantization configuration).
    Thresholds are then being calculated using the collected statistics and the model is quantized
    (both coefficients and activations by default).
    If a gptq configuration is passed, the quantized weights are optimized using knowledge
    distillation by comparing points between the float and quantized models, and minimizing the observed loss.

    Args:
        in_model: Model to quantize.
        representative_data_gen: Dataset used for calibration.
        n_iter: Number of calibration iterations to run.
        quant_config: QuantizationConfig containing parameters of how the model should be quantized. `Default configuration. <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10/model_compression_toolkit/common/quantization/quantization_config.py#L163>`_
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10/model_compression_toolkit/keras/default_framework_info.py#L114>`_
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        network_editor: List of EditRules. Each EditRule consists of a node filter and an action to change quantization settings of the filtered nodes.
        gptq_config: Configuration for using gradient-based PTQ (e.g. optimizer).
        analyze_similarity: Whether to plot similarity figures within TensorBoard (when logger is enabled) or not.
        target_kpi: KPI to constraint the search of the mixed-precision configuration for the model.

    Returns:
        A quantized model and information the user may need to handle the quantized model.

    """
    if quant_config.weights_bias_correction and gptq_config is not None:
        common.Logger.error('weights_bias_correction should be disabled in GPTQ mode')

    tb_w = _init_tensorboard_writer()

    tg = _prepare_model_for_quantization(in_model,
                                         representative_data_gen,
                                         network_editor,
                                         n_iter,
                                         quant_config,
                                         fw_info,
                                         tb_w,
                                         fw_impl)


    ######################################
    # Finalize bit widths
    ######################################
    if target_kpi is not None:
        assert isinstance(quant_config, MixedPrecisionQuantizationConfig)
        bit_widths_config = search_bit_width(tg,
                                             quant_config,
                                             fw_info,
                                             target_kpi,
                                             partial(fw_impl.get_sensitivity_evaluation_fn,
                                                     representative_data_gen=representative_data_gen,
                                                     fw_info=fw_info))
    else:
        bit_widths_config = None

    tg = set_bit_widths(quant_config,
                        tg,
                        fw_info,
                        bit_widths_config)

    quantized_model, user_info = _quantize_fixed_bit_widths_graph(analyze_similarity,
                                                                  fw_info,
                                                                  gptq_config,
                                                                  representative_data_gen,
                                                                  tb_w,
                                                                  tg,
                                                                  fw_impl)
    user_info.mixed_precision_cfg = bit_widths_config

    return quantized_model, user_info





def _init_tensorboard_writer() -> TensorboardWriter:
    """

    Returns: A TensorBoardWriter object initialized with the logger dir path if it was set, or None otherwise.

    """
    tb_w = None
    if common.Logger.LOG_PATH is not None:
        tb_log_dir = os.path.join(os.getcwd(), common.Logger.LOG_PATH, 'tensorboard_logs')
        common.Logger.info(f'To use Tensorboard, please run: tensorboard --logdir {tb_log_dir}')
        tb_w = TensorboardWriter(tb_log_dir)
    return tb_w


def _quantize_model(fw_info: FrameworkInfo,
                    tb_w: TensorboardWriter,
                    tg: Graph,
                    fw_impl: FrameworkImplementation) -> Tuple[Any, UserInformation]:
    """
    Quantize graph's weights, and build a quantized Keras model from it.

    Args:
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).
        tb_w: TensorBoardWriter object to log events.
        tg: A prepared for quantization graph.

    Returns:
        Quantize Keras model, and informat the user may need to use the quantized model.
    """

    quantized_tg = quantize_graph_weights(tg,
                                          fw_info=fw_info,
                                          fw_impl=fw_impl)
    if tb_w is not None:
        tb_w.add_graph(quantized_tg, 'after_quantization')

    quantized_graph_with_bias_correction = apply_bias_correction_to_graph(quantized_tg,
                                                                          fw_info=fw_info,
                                                                          fw_impl=fw_impl)
    if tb_w is not None:
        tb_w.add_graph(quantized_graph_with_bias_correction, 'after_bias_correction')

    ######################################
    # Back2Framework
    ######################################
    # Before building a quantized model, first apply some substitutions.
    quantized_graph_with_bias_correction = substitute(quantized_graph_with_bias_correction,
                                                      fw_impl.get_substitutions_pre_build())
    quantized_model, user_info = fw_impl.model_builder(quantized_graph_with_bias_correction,
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
        tg: Graph of quantized model.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).

    Returns:

    """
    if gptq_config is not None:
        common.Logger.info("Using experimental Gradient Based PTQ: If you encounter an issue "
                           "please file a bug. To disable it, do not pass a gptq configuration.")

        tg = fw_impl.gptq_training(tg,
                                   representative_data_gen,
                                   gptq_config,
                                   fw_info)

        if tb_w is not None:
            tb_w.add_graph(tg, 'after_gptq')
    return tg


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
    # Gradient Based Post Training Quantization
    #############################################
    tg = _apply_gptq(gptq_config,
                     representative_data_gen,
                     tb_w,
                     tg,
                     fw_info,
                     fw_impl)

    tg_float = copy.deepcopy(tg)  # Copy graph before quantization (for similarity analyzer)
    ######################################
    # Model Quantization
    ######################################
    quantized_model, user_info = _quantize_model(fw_info,
                                                 tb_w,
                                                 tg,
                                                 fw_impl)
    if analyze_similarity:
        _analyze_similarity(representative_data_gen,
                            tb_w,
                            tg,
                            tg_float)

    return quantized_model, user_info



def _prepare_model_for_quantization(in_model: Any,
                                    representative_data_gen: Callable,
                                    network_editor: List[EditRule] = [],
                                    n_iter: int = 500,
                                    quant_config: QuantizationConfig = DEFAULTCONFIG,
                                    fw_info: FrameworkInfo = None,
                                    tb_w: TensorboardWriter = None,
                                    fw_impl: FrameworkImplementation = None) -> Graph:
    """
    Prepare a trained Keras model for post-training quantization. The model is prepared to be quantized using a
    symmetric constraint quantization thresholds (power of two).
    The model is first read into a graph object and being optimized using several transformations (e.g.
    BatchNormalization folding to preceding layers). Then, using a given dataset, statistics (e.g. min/max,
    histogram, etc.) are being collected for each layer's output (and input, depends on the quantization configuration).
    Thresholds are then being calculated using the collected statistics. Finally, more transformations (based on
    statistics) are applied to increase model's performance.

    Args:
        in_model (Model): Keras model to optimize and prepare for quantization.
        representative_data_gen (Callable): Dataset used for calibration.
        network_editor (List[EditRule]): List of EditRules. Each EditRule consists of a node filter and an action to
        change quantization settings of the filtered nodes.
        n_iter (int): Number of calibration iterations to run.
        quant_config (QuantizationConfig): QuantizationConfig containing parameters of how the model should be
        quantized.
        fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g.,
        kernel channels indices, groups of layers by how they should be quantized, etc.)
        tb_w (TensorboardWriter): TensorboardWriter object to use for logging events such as graphs, histograms, etc.
        fw_impl (FrameworkImplementation): FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Graph object that represents the Keras model, contains thresholds, and ready for quantization.
    """

    ######################################
    # Represent model in a graph
    ######################################
    graph = fw_impl.model_reader(in_model)  # model reading

    if tb_w is not None:
        tb_w.add_graph(graph, 'initial_graph')

    ######################################
    # Graph substitution (pre statistics collection)
    ######################################
    transformed_graph = substitute(graph, fw_impl.get_substitutions_pre_statistics_collection())

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'pre_statistics_collection_substitutions')

    ######################################
    # Graph marking points
    ######################################
    transformed_graph = substitute(transformed_graph, fw_impl.get_substitutions_marking())

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_graph_marking')

    ######################################
    # Graph analyzing (attaching statistics collectors)
    ######################################
    analyzer_graph(fw_impl.attach_sc_to_node,
                   transformed_graph,
                   fw_info,
                   quant_config)  # Mark points for statistics collection

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_analyzer_graph')

    ######################################
    # Statistic collection
    ######################################
    mi = ModelCollector(transformed_graph,
                        fw_impl,
                        fw_info)

    for _ in tqdm(range(n_iter)):
        mi.infer(representative_data_gen())

    ######################################
    # Add quantization configurations
    ######################################
    transformed_graph = set_quantization_configuration_to_graph(transformed_graph,
                                                                quant_config,
                                                                fw_info)

    ######################################
    # Edit network according to user specific settings
    ######################################
    edit_network_graph(transformed_graph, fw_info, network_editor)

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
                                   fw_impl.get_substitutions_post_statistics_collection(quant_config))


    ######################################
    # Channel equalization
    ######################################
    transformed_graph = substitute(transformed_graph,
                                   fw_impl.get_substitutions_channel_equalization(quant_config,
                                                                                  fw_info))

    ######################################
    # Shift Negative Activations
    ######################################
    if quant_config.shift_negative_activation_correction:
        transformed_graph = fw_impl.shift_negative_correction(transformed_graph,
                                                                    quant_config,
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
