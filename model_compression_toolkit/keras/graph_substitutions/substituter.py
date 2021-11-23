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

from typing import List

from model_compression_toolkit import common
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig

from model_compression_toolkit.keras.graph_substitutions.substitutions.activation_decomposition import \
    ActivationDecomposition
from model_compression_toolkit.keras.graph_substitutions.substitutions.relu_bound_correction import \
    ReLUBoundCorrection
from model_compression_toolkit.keras.graph_substitutions.substitutions.batchnorm_folding import \
    BatchNormalizationFolding
from model_compression_toolkit.keras.graph_substitutions.substitutions.input_scaling import InputScaling, InputScalingWithPad
from model_compression_toolkit.keras.graph_substitutions.substitutions.mark_activation import MarkActivation
from model_compression_toolkit.keras.graph_substitutions.substitutions.remove_relu_upper_bound import \
    RemoveReLUUpperBound
from model_compression_toolkit.keras.graph_substitutions.substitutions.scale_equalization import \
    ScaleEqualization, ScaleEqualizationWithPad, \
    ScaleEqualizationMidActivation, ScaleEqualizationMidActivationWithPad
from model_compression_toolkit.keras.graph_substitutions.substitutions.separableconv_decomposition import \
    SeparableConvDecomposition
from model_compression_toolkit.keras.graph_substitutions.substitutions.shift_negative_activation import \
    apply_shift_negative_correction


def substitute(graph_to_substitute: common.Graph,
               substitutions_list: List[common.BaseSubstitution]) -> common.Graph:
    """
    Apply a list of substitutions on a graph.
    Args:
        graph: Graph to transform.
        substitutions_list: List of substitutions to apply on the graph.

    Returns:
        Transformed graph after applying all substitutions in substitutions_list.
    """

    graph = copy.deepcopy(graph_to_substitute)
    for substitution in substitutions_list:
        matched_nodes = graph.filter(substitution.matcher_instance)
        for idn in matched_nodes:
            graph = substitution.substitute(graph, idn)
    return graph


def graph_marking_substitute(graph: Graph) -> Graph:
    """
    Build a list of marking substitutions the graph should transformed according to (before statistics
    are being collected), apply these substitutions on the graph and return the transformed graph.
    Args:
        graph: Graph to apply substitutions on.

    Returns:
        Transformed graph after marking substitutions were applied.
    """

    marking_substitutions_list = [MarkActivation()]  # mark activation layers that their inputs should not be quantized
    return substitute(graph,
                      marking_substitutions_list)


def pre_statistics_collection_substitute(graph: Graph) -> Graph:
    """
    Build a list of substitutions the graph should transformed according to (before statistics
    are being collected), apply these substitutions on the graph and return the transformed graph.
    Args:
        graph: Graph to apply substitutions on.

    Returns:
        Transformed graph after substitutions.
    """
    substitutions_list = [SeparableConvDecomposition(),  # decompose separable node into depthwise and pointwise nodes
                          ActivationDecomposition(),  # extract activation from linear op to an additional layer
                          BatchNormalizationFolding(),  # fold batch normalization layer to the preceding linear layer
                          MarkActivation()]  # mark activation layers that their inputs should not be quantized

    return substitute(graph,
                      substitutions_list)


def post_statistics_collection_substitute(graph: Graph,
                                          quant_config: QuantizationConfig,
                                          fw_info: FrameworkInfo) -> Graph:
    """
    Build a list of substitutions the graph should transformed according to (after statistics
    were collected), apply these substitutions on the graph and return the transformed graph.

    Args:
        graph: Graph to apply substitutions on.
        quant_config: Quantization configuration to build the substitutions list according to.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)

    Returns:
        Transformed graph after substitutions.
    """
    substitutions_list = []
    ######################################
    # Scale Activations
    ######################################
    if quant_config.input_scaling:
        substitutions_list.append(InputScaling(quant_config,
                                               fw_info))
        substitutions_list.append(InputScalingWithPad(quant_config,
                                                      fw_info))

    ######################################
    # Scale Activations
    ######################################
    if quant_config.relu_unbound_correction:
        substitutions_list.append(ReLUBoundCorrection(quant_config,
                                                      fw_info))

    if quant_config.activation_channel_equalization:
        substitutions_list.append(ScaleEqualization(quant_config,
                                                    fw_info))
        substitutions_list.append(ScaleEqualizationWithPad(quant_config,
                                                           fw_info))
        substitutions_list.append(ScaleEqualizationMidActivation(quant_config,
                                                                 fw_info))
        substitutions_list.append(ScaleEqualizationMidActivationWithPad(quant_config,
                                                                        fw_info))

    ######################################
    # Shift Negative Activations
    ######################################
    if quant_config.shift_negative_activation_correction:
        graph = apply_shift_negative_correction(graph, quant_config, fw_info)

    return substitute(graph,
                      substitutions_list)


def pre_build_substitute(graph: Graph,
                         remove_relu_bound: bool = True) -> Graph:
    """
    Build a list of substitutions the graph should transformed according to (before building
    the model back from its graph), apply these substitutions on the graph and return the transformed graph.
    Args:
        graph: Graph to apply substitutions on.
        remove_relu_bound: Whether or not to remove bounds of bounded ReLUs in case the quantization threshold is
        bound the maximal value anyway.

    Returns:
        Transformed graph after substitutions.
    """
    substitutions_list = []
    if remove_relu_bound:
        substitutions_list.append(RemoveReLUUpperBound())

    return substitute(graph,
                      substitutions_list)
