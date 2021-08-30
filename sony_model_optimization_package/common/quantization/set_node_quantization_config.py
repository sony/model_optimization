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

from sony_model_optimization_package.common.framework_info import FrameworkInfo
from sony_model_optimization_package.common.graph.base_graph import Graph
from sony_model_optimization_package.common.quantization.node_quantization_config import create_node_activation_qc, \
    create_node_weights_qc
from sony_model_optimization_package.common.quantization.quantization_config import QuantizationConfig


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
