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
from typing import List

from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common import Graph, BaseNode, Logger
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_activations_computation \
    import \
    get_activations_qparams
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_weights_computation import \
    get_weights_qparams, get_channels_axis


def calculate_quantization_params(graph: Graph,
                                  fw_info: FrameworkInfo,
                                  nodes: List[BaseNode] = [],
                                  specific_nodes: bool = False,
                                  fw_impl: FrameworkImplementation = None):
    """
    For a graph, go over its nodes, compute quantization params (for both weights and activations according
    to the given framework info), and create and attach a NodeQuantizationConfig to each node (containing the
    computed params).
    By default, the function goes over all nodes in the graph. However, the specific_nodes flag enables
    to compute quantization paramss for specific nodes if the default behavior is unnecessary. For that,
    a list of nodes nodes should be passed as well.

    Args:
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        graph: Graph to compute its nodes' thresholds.
        nodes: List of nodes to compute their thresholds instead of computing it for all nodes in the graph.
        specific_nodes: Flag to compute thresholds for only specific nodes.
        fw_impl: FrameworkImplementation with specific framework implementations.

    """

    # Create a list of nodes to compute their thresholds
    nodes_list: List[BaseNode] = nodes if specific_nodes else graph.nodes()

    for n in nodes_list:  # iterate only nodes that we should compute their thresholds
        for candidate_qc in n.candidates_quantization_cfg:
            if n.is_weights_quantization_enabled():
                # If node's weights should be quantized, we compute its weights' quantization parameters
                output_channels_axis, _ = get_channels_axis(candidate_qc.weights_quantization_cfg, fw_info, n.type)
                weights_params = get_weights_qparams(n.get_weights_by_keys(fw_impl.constants.KERNEL),
                                                     candidate_qc.weights_quantization_cfg,
                                                     output_channels_axis)
                candidate_qc.weights_quantization_cfg.set_weights_quantization_param(weights_params)
                candidate_qc.weights_quantization_cfg.weights_channels_axis = output_channels_axis
            if n.is_activation_quantization_enabled():
                # If node's activations should be quantized as well, we compute its activation quantization parameters
                activation_params = get_activations_qparams(
                    activation_quant_cfg=candidate_qc.activation_quantization_cfg,
                    nodes_prior_info=n.prior_info,
                    out_stats_container=graph.get_out_stats_collector(n))
                # Create a NodeQuantizationConfig containing all quantization params and attach it to the node
                candidate_qc.activation_quantization_cfg.set_activation_quantization_param(activation_params)
