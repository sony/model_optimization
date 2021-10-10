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
from typing import Callable, List, Tuple

from tensorflow.keras.models import Model
from tqdm import tqdm

from model_compression_toolkit import common
from model_compression_toolkit.common import FrameworkInfo
from model_compression_toolkit.common.constants import NUM_SAMPLES_CS_TENSORBOARD
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.network_editors.actions import EditRule
from model_compression_toolkit.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.common.quantization.quantization_config import DEFAULTCONFIG
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.quantize_model import calculate_quantization_params
from model_compression_toolkit.common.quantization.set_node_quantization_config import set_qcs_to_graph_nodes
from model_compression_toolkit.keras.visualization.nn_visualizer import KerasNNVisualizer
from model_compression_toolkit.keras.back2framework.model_builder import model_builder, ModelBuilderMode
from model_compression_toolkit.keras.back2framework.model_collector import ModelCollector
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.keras.graph_substitutions.substituter import graph_marking_substitute
from model_compression_toolkit.keras.graph_substitutions.substituter import post_statistics_collection_substitute
from model_compression_toolkit.keras.graph_substitutions.substituter import pre_statistics_collection_substitute
from model_compression_toolkit.keras.knowledge_distillation.training_wrapper import KnowledgeDistillationConfig
from model_compression_toolkit.keras.knowledge_distillation.training_wrapper import \
    knowledge_distillation_training_wrapper
from model_compression_toolkit.keras.reader.reader import model_reader
from model_compression_toolkit.keras.tensor_marking import get_node_stats_collector
from model_compression_toolkit.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.keras.node_to_bops import node_to_bops


def _post_training_model_builder(in_model: Model,
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

    is_logger_path_set = common.Logger.LOG_PATH is not None

    ######################################
    # Represent model in a graph
    ######################################
    graph = model_reader(in_model)  # model reading

    if is_logger_path_set:
        tb_w.add_graph(graph, 'initial_graph', node_to_bops)

    ######################################
    # Graph substitution (pre statistics collection)
    ######################################
    transformed_graph = pre_statistics_collection_substitute(graph)
    if is_logger_path_set:
        tb_w.add_graph(transformed_graph, 'pre_statistics_collection_substitutions', node_to_bops)

    ######################################
    # Graph marking points
    ######################################
    transformed_graph = graph_marking_substitute(transformed_graph)

    if is_logger_path_set:
        tb_w.add_graph(transformed_graph, 'after_graph_marking', node_to_bops)

    ######################################
    # Graph analyzing (attaching statistics collectors)
    ######################################
    analyzer_graph(get_node_stats_collector,
                   transformed_graph,
                   fw_info,
                   quant_config)  # Mark points for statistics collection

    if is_logger_path_set:
        tb_w.add_graph(transformed_graph, 'after_analyzer_graph', node_to_bops)

    ######################################
    # Statistic collection
    ######################################
    mi = ModelCollector(transformed_graph)
    for _ in tqdm(range(n_iter)):
        mi.infer(representative_data_gen())

    ###########################################
    # Add quantization configurations to nodes
    ############################################
    transformed_graph = set_qcs_to_graph_nodes(transformed_graph,
                                               quant_config,
                                               fw_info)

    ######################################
    # Edit network according to user specific settings
    ######################################
    edit_network_graph(transformed_graph, fw_info, network_editor)

    ######################################
    # Calculate quantization params
    ######################################
    calculate_quantization_params(transformed_graph, fw_info)

    if is_logger_path_set:
        tb_w.add_graph(transformed_graph, 'thresholds_selection', node_to_bops)
        tb_w.add_all_statistics(transformed_graph, 'thresholds_selection')

    ######################################
    # Graph substitution (post statistics collection)
    ######################################
    transformed_graph = post_statistics_collection_substitute(transformed_graph,
                                                              quant_config,
                                                              fw_info)

    if is_logger_path_set:
        tb_w.add_graph(transformed_graph, 'post_statistics_collection_substitutions', node_to_bops)
        tb_w.add_all_statistics(transformed_graph, 'post_statistics_collection_substitutions')

    return transformed_graph


def keras_post_training_quantization(in_model: Model,
                                     representative_data_gen: Callable,
                                     n_iter: int = 500,
                                     quant_config: QuantizationConfig = DEFAULTCONFIG,
                                     fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                                     network_editor: List[EditRule] = [],
                                     knowledge_distillation_config: KnowledgeDistillationConfig = None,
                                     analyze_similarity: bool = False):
    """
    Quantize a trained Keras model using post-training quantization. The model is quantized using a
    symmetric constraint quantization thresholds (power of two).
    The model is first optimized using several transformations (e.g. BatchNormalization folding to
    preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
    being collected for each layer's output (and input, depends on the quantization configuration).
    Thresholds are then being calculated using the collected statistics and the model is quantized
    (both coefficients and activations by default).
    If a knowledge distillation configuration is passed, the quantized weights are optimized using knowledge
    distillation by comparing points between the float and quantized models, and minimizing the observed loss.

    Args:
        in_model (Model): Keras model to quantize.
        representative_data_gen (Callable): Dataset used for calibration.
        n_iter (int): Number of calibration iterations to run.
        quant_config (QuantizationConfig): QuantizationConfig containing parameters of how the model should be quantized. `Default configuration. <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10/model_compression_toolkit/common/quantization/quantization_config.py#L163>`_
        fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10/model_compression_toolkit/keras/default_framework_info.py#L114>`_
        network_editor (List[EditRule]): List of EditRules. Each EditRule consists of a node filter and an action to change quantization settings of the filtered nodes.
        knowledge_distillation_config (KnowledgeDistillationConfig): Configuration for using knowledge distillation (e.g. optimizer).
        analyze_similarity (bool): Whether to plot similarity figures within TensorBoard (when logger is enabled) or not.

    Returns:
        A quantized model.

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

    is_logger_path_set = common.Logger.LOG_PATH is not None

    tb_w = None
    if is_logger_path_set:
        tb_log_dir = os.path.join(os.getcwd(), common.Logger.LOG_PATH, 'tensorboard_logs')
        print(f'To use Tensorboard, please run: tensorboard --logdir {tb_log_dir}')
        tb_w = TensorboardWriter(tb_log_dir)

    tg = _post_training_model_builder(in_model,
                                      representative_data_gen,
                                      network_editor,
                                      n_iter,
                                      quant_config,
                                      fw_info,
                                      tb_w)

    ######################################
    # Knowledge Distillation
    ######################################
    if knowledge_distillation_config is not None:
        print("Using experimental knowledge distillation quantization: If you encounter an issue "
              "please file a bug. To disable it, do not pass a knowledge-distillation configuration.")

        tg = knowledge_distillation_training_wrapper(tg,
                                                     representative_data_gen,
                                                     knowledge_distillation_config,
                                                     fw_info)
        if is_logger_path_set:
            tb_w.add_graph(tg, 'after_kd', node_to_bops)

    tg_float = copy.deepcopy(tg)  # Copy graph before quantization
    ######################################
    # Model Quantization
    ######################################
    common.quantize_model(tg,
                          fw_info=fw_info)
    if is_logger_path_set:
        tb_w.add_graph(tg, 'after_quantization', node_to_bops)

    ######################################
    # Back2Framework
    ######################################
    quantized_model = model_builder(tg,
                                    mode=ModelBuilderMode.QUANTIZED)

    if is_logger_path_set and analyze_similarity:
        visual = KerasNNVisualizer(tg_float, tg)
        for i in range(NUM_SAMPLES_CS_TENSORBOARD):
            figure = visual.plot_cs_graph(representative_data_gen())
            tb_w.add_figure(figure, f'cosine_similarity_sample_{i}')
        tb_w.close()

    return quantized_model
