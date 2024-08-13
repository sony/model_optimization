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


from typing import Callable, Any

from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.fusion.layer_fusing import fusion
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.quantization.bit_width_config import BitWidthConfig
from model_compression_toolkit.core.common.quantization.filter_nodes_candidates import filter_nodes_candidates
from model_compression_toolkit.core.common.quantization.quantization_config import DEFAULTCONFIG
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.common.substitutions.linear_collapsing_substitution import \
    linear_collapsing_substitute
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter


def graph_preparation_runner(in_model: Any,
                             representative_data_gen: Callable,
                             quantization_config: QuantizationConfig,
                             fw_info: FrameworkInfo,
                             fw_impl: FrameworkImplementation,
                             tpc: TargetPlatformCapabilities,
                             bit_width_config: BitWidthConfig = None,
                             tb_w: TensorboardWriter = None,
                             mixed_precision_enable: bool = False,
                             running_gptq: bool = False) -> Graph:
    """
    Runs all required preparations in order to build a quantization graph from the given model,
    quantization configuration and target platform specifications.
    This runner include the following steps:
        - Reading and building a graph from the given model.
        - Setting quantization config to each relevant node in the graph.
        - Apply all necessary substitutions to finalize the graph for quantization.

    Args:
        in_model (Any): Model to quantize.
        representative_data_gen (Callable): Dataset used for calibration.
        quantization_config (QuantizationConfig): QuantizationConfig containing parameters of how the model should be quantized.
        fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.).
        fw_impl (FrameworkImplementation): FrameworkImplementation object with a specific framework methods implementation.
        tpc (TargetPlatformCapabilities): TargetPlatformCapabilities object that models the inference target platform and
            the attached framework operator's information.
        bit_width_config (BitWidthConfig): Config for bit-width selection. Defaults to None.
        tb_w (TensorboardWriter): TensorboardWriter object for logging.
        mixed_precision_enable (bool): is mixed precision enabled.
        running_gptq (bool): Whether or not a GPTQ optimization is planned to run after the PTQ process.

    Returns:
        An internal graph representation of the input model.
    """

    graph = read_model_to_graph(in_model,
                                representative_data_gen,
                                tpc,
                                fw_info,
                                fw_impl)

    if tb_w is not None:
        tb_w.add_graph(graph, 'initial_graph')

    transformed_graph = get_finalized_graph(graph,
                                            tpc,
                                            quantization_config,
                                            bit_width_config,
                                            fw_info,
                                            tb_w,
                                            fw_impl,
                                            mixed_precision_enable=mixed_precision_enable,
                                            running_gptq=running_gptq)

    return transformed_graph


def get_finalized_graph(initial_graph: Graph,
                        tpc: TargetPlatformCapabilities,
                        quant_config: QuantizationConfig = DEFAULTCONFIG,
                        bit_width_config: BitWidthConfig = None,
                        fw_info: FrameworkInfo = None,
                        tb_w: TensorboardWriter = None,
                        fw_impl: FrameworkImplementation = None,
                        mixed_precision_enable: bool = False,
                        running_gptq: bool = False) -> Graph:
    """
    Applies all edit operation (edit, substitutions, etc.) on the model's graph, to prepare it for the quantization
    process. All future graph substitutions and operations that change the graph should be added to this method.

    Args:
        initial_graph (Graph): Graph to apply the changes to.
        tpc (TargetPlatformCapabilities): TargetPlatformCapabilities object that describes the desired inference target platform (includes fusing patterns MCT should handle).
        quant_config (QuantizationConfig): QuantizationConfig containing parameters of how the model should be
            quantized.
        bit_width_config (BitWidthConfig): Config for bit-width selection. Defaults to None.
        fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g.,
            kernel channels indices, groups of layers by how they should be quantized, etc.)
        tb_w (TensorboardWriter): TensorboardWriter object to use for logging events such as graphs, histograms, etc.
        fw_impl (FrameworkImplementation): FrameworkImplementation object with a specific framework methods implementation.
        mixed_precision_enable: is mixed precision enabled.
        running_gptq: Whether or not a GPTQ optimization is planned to run after the PTQ process.

    Returns: Graph object that represents the model, after applying all required modifications to it.
    """

    ######################################
    # Graph substitution (prepare graph)
    ######################################
    graph = substitute(initial_graph, fw_impl.get_substitutions_prepare_graph(fw_info))

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
        transformed_graph = linear_collapsing_substitute(transformed_graph, fw_impl.get_op2d_add_const_collapsing_substitution())
    if quant_config.residual_collapsing:
        transformed_graph = substitute(transformed_graph, fw_impl.get_residual_collapsing_substitution())

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'pre_statistics_collection_substitutions')

    ######################################
    # Add quantization configurations
    ######################################
    transformed_graph = set_quantization_configuration_to_graph(graph=transformed_graph,
                                                                quant_config=quant_config,
                                                                bit_width_config=bit_width_config,
                                                                mixed_precision_enable=mixed_precision_enable,
                                                                running_gptq=running_gptq)

    ######################################
    # Layer fusing
    ######################################
    transformed_graph = fusion(transformed_graph, tpc)

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
