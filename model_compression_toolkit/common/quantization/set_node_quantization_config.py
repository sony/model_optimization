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

from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common import Logger
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.quantization.node_quantization_config import NodeActivationQuantizationConfig, \
    NodeWeightsQuantizationConfig
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.quantization_params_fn_selection import \
    get_activation_quantization_params_fn, get_weights_quantization_params_fn


def set_quantization_configuration_to_graph(graph: Graph,
                                            quant_config: QuantizationConfig,
                                            fw_info: FrameworkInfo) -> Graph:
    """
    Add quantization configuration for each graph node.

    Args:
        graph: Graph for which to add quantization info to each node.
        quant_config: Quantization configuration containing parameters for how the graph should be quantized.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)

    Returns:
        The graph with quantization configurations attached to each node in it.
    """

    graph_with_qcs = copy.deepcopy(graph)

    for n in graph_with_qcs.nodes:
        # Set qc only when needed
        quantize_node_weights = False
        quantize_node_activations = False

        if fw_info.in_kernel_ops(n):
            quantize_node_weights = True
            quantize_node_activations = n.output_quantization
        elif fw_info.in_activation_ops(n):
            quantize_node_activations = True

        if quantize_node_activations:
            # Create activation QC for this node
            out_sc = graph_with_qcs.get_out_stats_collector(n)
            sc = out_sc[0] if isinstance(out_sc, list) else out_sc
            use_min_max = sc.use_min_max
            n.activation_quantization_cfg = create_node_activation_qc(quant_config,
                                                                      fw_info,
                                                                      use_min_max)
        if quantize_node_weights:
            # Create weights QC for this node
            weight_channel_axis = fw_info.kernel_channels_mapping.get(n.layer_class)[0]
            n.candidates_weights_quantization_cfg = _create_node_candidates_weights_qc(quant_config,
                                                                                      fw_info,
                                                                                      weight_channel_axis)
    return graph_with_qcs


def create_node_activation_qc(qc: QuantizationConfig,
                              fw_info: FrameworkInfo,
                              use_min_max: bool) -> NodeActivationQuantizationConfig:
    """
    Create a activations quantization configuration from a QuantizationConfig object.

    Args:
        qc: QuantizationConfig to create the node's config from.
        fw_info: Information about the specific framework the node was created from (e.g., whether or not its
        weights/activations should be quantized)
        use_min_max: Whether the collected min/max statistics should be used when the threshold is computed or not.

    Returns:
        Activation quantization configuration of a node.
    """

    activation_quantization_fn = fw_info.activation_quantizer_mapping.get(qc.activation_quantization_method)
    if activation_quantization_fn is None:
        Logger.critical('Unknown quantization method for activations')

    activation_quantization_params_fn = get_activation_quantization_params_fn(qc.activation_quantization_method,
                                                                              qc.activation_threshold_method,
                                                                              use_min_max)

    return NodeActivationQuantizationConfig(qc,
                                            activation_quantization_fn,
                                            activation_quantization_params_fn)


def create_node_weights_qc(qc: QuantizationConfig,
                           fw_info: FrameworkInfo,
                           weight_channel_axis: int) -> NodeWeightsQuantizationConfig:
    """
    Create a weights quantization configuration from a QuantizationConfig object.

    Args:
        qc: QuantizationConfig to create the node's config from.
        fw_info: Information about the specific framework the node was created from (e.g., whether or not its
        weights/activations should be quantized)
        weight_channel_axis: Axis to quantize a node's kernel when quantizing per-channel.

    Returns:
        Weights quantization configuration of a node.
    """

    weights_quantization_fn = fw_info.weights_quantizer_mapping.get(qc.weights_quantization_method)

    if weights_quantization_fn is None:
        Logger.critical('Unknown quantization method for weights')

    weights_quantization_params_fn = get_weights_quantization_params_fn(qc.weights_quantization_method,
                                                                        qc.weights_threshold_method)

    return NodeWeightsQuantizationConfig(qc,
                                         weights_quantization_fn,
                                         weights_quantization_params_fn,
                                         weight_channel_axis)



def _create_node_candidates_weights_qc(qc: QuantizationConfig,
                                      fw_info: FrameworkInfo,
                                      weight_channel_axis: int) -> List[NodeWeightsQuantizationConfig]:
    """
    Create a list of candidates of weights quantization configurations for a node.

    Args:
        qc: Quantization configuration the quantization process should follow.
        fw_info: Framework information (e.g., which layers should have their kernels' quantized).
        weight_channel_axis: Output channel index of the node's kernel.

    Returns:
        List of candidates of weights quantization configurations to set for a node.
    """

    candidats = []
    if isinstance(qc, MixedPrecisionQuantizationConfig):
        qc.weights_n_bits.sort(reverse=True)
        for nbits in qc.weights_n_bits:
            single_nbits_qc = copy.deepcopy(qc)
            single_nbits_qc.weights_n_bits = nbits
            candidats.append(create_node_weights_qc(single_nbits_qc, fw_info, weight_channel_axis))
    else:
        candidats.append(create_node_weights_qc(qc, fw_info, weight_channel_axis))

    return candidats