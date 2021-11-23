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
from typing import Callable, List, Tuple

from tensorflow.keras.models import Model
from tqdm import tqdm

from model_compression_toolkit import common
from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common import FrameworkInfo
from model_compression_toolkit.common.constants import NUM_SAMPLES_CS_TENSORBOARD
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.mixed_precision.bit_width_setter import set_bit_widths

from model_compression_toolkit.common.mixed_precision.mixed_precision_search_facade import search_bit_width
from model_compression_toolkit.common.network_editors.actions import EditRule
from model_compression_toolkit.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig, DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.common.bias_correction.compute_bias_correction_of_graph import compute_bias_correction_of_graph

from model_compression_toolkit.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.common.quantization.quantization_config import DEFAULTCONFIG
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params

from model_compression_toolkit.common.quantization.set_node_quantization_config import set_quantization_configuration_to_graph
from model_compression_toolkit.common.user_info import UserInformation
from model_compression_toolkit.keras.back2framework.model_builder import model_builder, ModelBuilderMode
from model_compression_toolkit.keras.back2framework.model_collector import ModelCollector
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.keras.graph_substitutions.substituter import graph_marking_substitute
from model_compression_toolkit.keras.graph_substitutions.substituter import post_statistics_collection_substitute
from model_compression_toolkit.keras.graph_substitutions.substituter import pre_statistics_collection_substitute
from model_compression_toolkit.keras.gradient_ptq.training_wrapper import GradientPTQConfig
from model_compression_toolkit.keras.gradient_ptq.training_wrapper import gptq_training_wrapper
from model_compression_toolkit.keras.mixed_precision.sensitivity_evaluation import get_sensitivity_evaluation
from model_compression_toolkit.keras.reader.reader import model_reader
from model_compression_toolkit.keras.tensor_marking import get_node_stats_collector
from model_compression_toolkit.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.common.bias_correction.apply_bias_correction_to_graph import apply_bias_correction_to_graph
from model_compression_toolkit.keras.visualization.nn_visualizer import KerasNNVisualizer


def _prepare_model_for_quantization(in_model: Model,
                                    representative_data_gen: Callable,
                                    network_editor: List[EditRule] = [],
                                    n_iter: int = 500,
                                    quant_config: QuantizationConfig = DEFAULTCONFIG,
                                    fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                                    tb_w: TensorboardWriter = None) -> Graph:
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

    Returns:
        Graph object that represents the Keras model, contains thresholds, and ready for quantization.
    """

    ######################################
    # Represent model in a graph
    ######################################
    graph = model_reader(in_model)  # model reading

    if tb_w is not None:
        tb_w.add_graph(graph, 'initial_graph')

    ######################################
    # Graph substitution (pre statistics collection)
    ######################################
    transformed_graph = pre_statistics_collection_substitute(graph)

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'pre_statistics_collection_substitutions')

    ######################################
    # Graph marking points
    ######################################
    transformed_graph = graph_marking_substitute(transformed_graph)

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_graph_marking')

    ######################################
    # Graph analyzing (attaching statistics collectors)
    ######################################
    analyzer_graph(get_node_stats_collector,
                   transformed_graph,
                   fw_info,
                   quant_config)  # Mark points for statistics collection

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_analyzer_graph')

    ######################################
    # Statistic collection
    ######################################
    mi = ModelCollector(transformed_graph)
    for _ in tqdm(range(n_iter)):
        mi.infer(representative_data_gen())

    ######################################
    # Add quantization configurations
    ######################################
    transformed_graph = set_quantization_configuration_to_graph(transformed_graph, quant_config, fw_info)

    ######################################
    # Edit network according to user specific settings
    ######################################
    edit_network_graph(transformed_graph, fw_info, network_editor)

    ######################################
    # Calculate quantization params
    ######################################
    calculate_quantization_params(transformed_graph, fw_info)

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'thresholds_selection')
        tb_w.add_all_statistics(transformed_graph, 'thresholds_selection')

    ######################################
    # Graph substitution (post statistics collection)
    ######################################
    transformed_graph = post_statistics_collection_substitute(transformed_graph,
                                                              quant_config,
                                                              fw_info)

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'post_statistics_collection_substitutions')
        tb_w.add_all_statistics(transformed_graph, 'post_statistics_collection_substitutions')

    ########################################################
    # Compute bias correction to nodes' config candidates
    ########################################################
    tg_with_bias = compute_bias_correction_of_graph(transformed_graph, fw_info)

    if tb_w is not None:
        tb_w.add_graph(tg_with_bias, 'bias_correction_computation')

    for n in tg_with_bias.nodes:
        assert n.final_weights_quantization_cfg is None

    return tg_with_bias


def keras_post_training_quantization(in_model: Model,
                                     representative_data_gen: Callable,
                                     n_iter: int = 500,
                                     quant_config: QuantizationConfig = DEFAULTCONFIG,
                                     fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                                     network_editor: List[EditRule] = [],
                                     gptq_config: GradientPTQConfig = None,
                                     analyze_similarity: bool = False):
    """
    Quantize a trained Keras model using post-training quantization. The model is quantized using a
    symmetric constraint quantization thresholds (power of two).
    The model is first optimized using several transformations (e.g. BatchNormalization folding to
    preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
    being collected for each layer's output (and input, depends on the quantization configuration).
    Thresholds are then being calculated using the collected statistics and the model is quantized
    (both coefficients and activations by default).
    If a gptq configuration is passed, the quantized weights are optimized using gradient based post
    training quantization by comparing points between the float and quantized models, and minimizing the observed loss.

    Args:
        in_model (Model): Keras model to quantize.
        representative_data_gen (Callable): Dataset used for calibration.
        n_iter (int): Number of calibration iterations to run.
        quant_config (QuantizationConfig): QuantizationConfig containing parameters of how the model should be quantized. `Default configuration. <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10/model_compression_toolkit/common/quantization/quantization_config.py#L154>`_
        fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10/model_compression_toolkit/keras/default_framework_info.py#L113>`_
        network_editor (List[EditRule]): List of EditRules. Each EditRule consists of a node filter and an action to change quantization settings of the filtered nodes.
        gptq_config (GradientPTQConfig): Configuration for using gptq (e.g. optimizer).
        analyze_similarity (bool): Whether to plot similarity figures within TensorBoard (when logger is enabled) or not.

    Returns:
        A quantized model and information the user may need to handle the quantized model.

    Examples:
        Import a Keras model:

        >>> from tensorflow.keras.applications.mobilenet import MobileNet
        >>> model = MobileNet()

        Create a random dataset generator:

        >>> import numpy as np
        >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

        Import mct and pass the model with the representative dataset generator to get a quantized model:

        >>> import model_compression_toolkit as mct
        >>> quantized_model, quantization_info = mct.keras_post_training_quantization(model, repr_datagen)

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
                                         tb_w)

    ######################################
    # Finalize bit widths
    ######################################
    tg = set_bit_widths(quant_config,
                        tg,
                        fw_info)

    quantized_model, user_info = _quantize_fixed_bit_widths_graph(analyze_similarity,
                                                                  fw_info,
                                                                  gptq_config,
                                                                  representative_data_gen,
                                                                  tb_w,
                                                                  tg)

    return quantized_model, user_info


def keras_post_training_quantization_mixed_precision(in_model: Model,
                                                     representative_data_gen: Callable,
                                                     n_iter: int = 500,
                                                     quant_config: MixedPrecisionQuantizationConfig = DEFAULT_MIXEDPRECISION_CONFIG,
                                                     fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                                                     network_editor: List[EditRule] = [],
                                                     gptq_config: GradientPTQConfig = None,
                                                     bit_widths_config: List[int] = None,
                                                     analyze_similarity: bool = False,
                                                     target_kpi: KPI = None):
    """
     Quantize a trained Keras model using post-training quantization. The model is quantized using a
     symmetric constraint quantization thresholds (power of two).
     The model is first optimized using several transformations (e.g. BatchNormalization folding to
     preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
     being collected for each layer's output (and input, depends on the quantization configuration).
     For each possible bit width (per layer) a threshold is then being calculated using the collected
     statistics. Then, using an ILP solver we find a mixed-precision configuration, and set a bit width
     for each layer. The model is then quantized (both coefficients and activations by default).
     In order to limit the maximal model's size, a target KPI can be passed after weights_memory
     is set (in bytes).
     For now, mixed precision is supported for weights only.
     If a gptq configuration is passed, the quantized weights are optimized using gradient based post
     training quantization by comparing points between the float and quantized models, and minimizing the observed loss.
     Notice that this feature is experimental.

     Args:
         in_model (Model): Keras model to quantize.
         representative_data_gen (Callable): Dataset used for calibration.
         n_iter (int): Number of calibration iterations to run.
         quant_config (MixedPrecisionQuantizationConfig): QuantizationConfig containing parameters of how the model should be quantized.
         fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/keras/default_framework_info.py#L100>`_
         network_editor (List[EditRule]): List of EditRules. Each EditRule consists of a node filter and an action to change quantization settings of the filtered nodes.
         gptq_config (GradientPTQConfig): Configuration for using GPTQ (e.g. optimizer).
         bit_widths_config (List[int]): Mixed-precision configuration to set bit widths for different layers.
         analyze_similarity (bool): Whether to plot similarity figures within TensorBoard (when logger is enabled) or not.
         target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.

     Returns:
         A quantized model and information the user may need to handle the quantized model.

     Examples:
         Import MCT:

         >>> import model_compression_toolkit as mct

         Import a Keras model:

         >>> from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
         >>> model = MobileNetV2()

         Create a random dataset generator:

         >>> import numpy as np
         >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

         Create a mixed-precision configuration, to quantize a model with different bitwidths for different layers.
         Here, each layer can be quantized by 2, 4 or 8 bits:

         >>> config = mct.MixedPrecisionQuantizationConfig(weights_n_bits=[4, 2, 8])

         Create a KPI object to limit our returned model's size. Note that this value affects only coefficients that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value, while the bias will not):

         >>> kpi = mct.KPI(model.count_params() * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

         Pass the model, the representative dataset generator, the configuration and the target KPI to get a quantized model:

         >>> quantized_model, quantization_info = mct.keras_post_training_quantization_mixed_precision(model, repr_datagen, n_iter=10, quant_config=config, target_kpi=kpi)

         For more configuration options, please take a look at our `API documentation <https://sony.github.io/model_optimization/api/api_docs/modules/mixed_precision_quantization_config.html>`_.

     """

    if quant_config.weights_bias_correction and gptq_config is not None:
        common.Logger.error('weights_bias_correction should be disabled in GPTQ mode')

    common.Logger.info("Using experimental mixed-precision quantization. "
                       "If you encounter an issue please file a bug.")

    if target_kpi is None:
        common.Logger.warning("No KPI was passed. Using non mixed-precision compression process...")
        # Before starting non-mixed-precision process, we need to set only single bit width, so we take the best
        # option which is the maximal number of bits.
        quant_config.weights_n_bits = [max(quant_config.weights_n_bits)]
        return keras_post_training_quantization(in_model,
                                                representative_data_gen,
                                                n_iter,
                                                quant_config,
                                                fw_info,
                                                network_editor,
                                                gptq_config,
                                                analyze_similarity)

    tb_w = _init_tensorboard_writer()

    tg = _prepare_model_for_quantization(in_model,
                                         representative_data_gen,
                                         network_editor,
                                         n_iter,
                                         quant_config,
                                         fw_info,
                                         tb_w)

    ######################################
    # Finalize bit widths
    ######################################

    if bit_widths_config is None:
        bit_widths_config = search_bit_width(tg,
                                             quant_config,
                                             fw_info,
                                             target_kpi,
                                             partial(get_sensitivity_evaluation,
                                                     representative_data_gen=representative_data_gen,
                                                     fw_info=fw_info))

    tg = set_bit_widths(quant_config,
                        tg,
                        fw_info,
                        bit_widths_config)

    quantized_model, user_info = _quantize_fixed_bit_widths_graph(analyze_similarity,
                                                                  fw_info,
                                                                  gptq_config,
                                                                  representative_data_gen,
                                                                  tb_w,
                                                                  tg)
    user_info.mixed_precision_cfg = bit_widths_config

    return quantized_model, user_info


def _quantize_fixed_bit_widths_graph(analyze_similarity: bool,
                                     fw_info: FrameworkInfo,
                                     gptq_config: GradientPTQConfig,
                                     representative_data_gen: Callable,
                                     tb_w: TensorboardWriter,
                                     tg: Graph) -> Tuple[Model, UserInformation]:
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
                     fw_info)

    tg_float = copy.deepcopy(tg)  # Copy graph before quantization (for similarity analyzer)
    ######################################
    # Model Quantization
    ######################################
    quantized_model, user_info = _quantize_model(fw_info,
                                                 tb_w,
                                                 tg)
    if analyze_similarity:
        _analyze_similarity(representative_data_gen,
                            tb_w,
                            tg,
                            tg_float)

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
                    tg: Graph) -> Tuple[Model, UserInformation]:
    """
    Quantize graph's weights, and build a quantized Keras model from it.

    Args:
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).
        tb_w: TensorBoardWriter object to log events.
        tg: A prepared for quantization graph.

    Returns:
        Quantize Keras model, and informat the user may need to use the quantized model.
    """

    quantized_tg = quantize_graph_weights(tg, fw_info=fw_info)
    if tb_w is not None:
        tb_w.add_graph(quantized_tg, 'after_quantization')

    quantized_graph_with_bias_correction = apply_bias_correction_to_graph(quantized_tg,
                                                                          fw_info=fw_info)
    if tb_w is not None:
        tb_w.add_graph(quantized_graph_with_bias_correction, 'after_bias_correction')

    ######################################
    # Back2Framework
    ######################################
    quantized_model, user_info = model_builder(quantized_graph_with_bias_correction,
                                               mode=ModelBuilderMode.QUANTIZED)

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
                fw_info: FrameworkInfo) -> Graph:
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

        tg = gptq_training_wrapper(tg,
                                   representative_data_gen,
                                   gptq_config,
                                   fw_info)

        if tb_w is not None:
            tb_w.add_graph(tg, 'after_gptq')
    return tg
