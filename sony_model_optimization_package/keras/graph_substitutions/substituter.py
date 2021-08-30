# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


import copy

from typing import List

from sony_model_optimization_package import common
from sony_model_optimization_package.common.framework_info import FrameworkInfo
from sony_model_optimization_package.common.graph.base_graph import Graph
from sony_model_optimization_package.common.quantization.quantization_config import QuantizationConfig

from sony_model_optimization_package.keras.graph_substitutions.substitutions.activation_decomposition import \
    ActivationDecomposition
from sony_model_optimization_package.keras.graph_substitutions.substitutions.relu_bound_correction import \
    ReLUBoundCorrection
from sony_model_optimization_package.keras.graph_substitutions.substitutions.batchnorm_folding import \
    BatchNormalizationFolding
from sony_model_optimization_package.keras.graph_substitutions.substitutions.input_scaling import InputScaling, InputScalingWithPad
from sony_model_optimization_package.keras.graph_substitutions.substitutions.mark_activation import MarkActivation
from sony_model_optimization_package.keras.graph_substitutions.substitutions.remove_relu_upper_bound import \
    RemoveReLUUpperBound
from sony_model_optimization_package.keras.graph_substitutions.substitutions.scale_equalization import \
    ScaleEqualization, ScaleEqualizationWithPad, \
    ScaleEqualizationMidActivation, ScaleEqualizationMidActivationWithPad
from sony_model_optimization_package.keras.graph_substitutions.substitutions.separableconv_decomposition import \
    SeparableConvDecomposition
from sony_model_optimization_package.keras.graph_substitutions.substitutions.shift_negative_activation import \
    apply_shift_negative_correction


def substitute(graph: common.Graph,
               substitutions_list: List[common.BaseSubstitution]) -> common.Graph:
    """
    Apply a list of substitutions on a graph.
    Args:
        graph: Graph to transform.
        substitutions_list: List of substitutions to apply on the graph.

    Returns:
        Transformed graph after applying all substitutions in substitutions_list.
    """

    # graph = copy.deepcopy(graph)
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
