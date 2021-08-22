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

from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.quantization.node_quantization_config import create_node_activation_qc, \
    create_node_weights_qc
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig


def set_qcs_to_graph_nodes(graph: Graph,
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
            n.weights_quantization_cfg = create_node_weights_qc(quant_config,
                                                                fw_info,
                                                                weight_channel_axis)
    return graph_with_qcs
