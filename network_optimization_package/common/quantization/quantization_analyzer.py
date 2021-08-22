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

from typing import Callable

from networkx.algorithms.dag import topological_sort

from network_optimization_package import common


def create_tensor2node(graph: common.Graph,
                       node: common.Node):
    """
    Force tensor creation and assignment for a node.
    Args:
        graph: Graph of the node (for retrieving the current tensor).
        node: Node to create a tensor for.

    """
    current_tensor = graph.get_out_stats_collector(node)
    if isinstance(current_tensor, common.NoStatsContainer) or current_tensor is None:
        graph.set_out_stats_collector_to_node(node, common.StatsContainer())


def analyzer_graph(node_analyze_func: Callable,
                   graph: common.Graph,
                   fw_info: common.FrameworkInfo,
                   qc: common.QuantizationConfig = common.DEFAULTCONFIG):
    """
    Go over all nodes in a graph, and create and set statistics collection tensors for each node's input and output.
    The tensors are stored in the graph.
    The kind of tensor that is created for each node is determined according to:
    node_analyze_func, groups mapping (operator to quantization treatment), and the overall quantization configuration.

    Args:
        fw_info: Information relevant to a specific framework about how layers should be quantized.
        node_analyze_func: Function which returns a tensor for statistics collection by a node.
        graph: Graph to set its tensors.
        qc: Quantization configuration containing parameters for how the graph should be quantized.

    """
    nodes_sorted = topological_sort(graph)
    for n in nodes_sorted:
        t = node_analyze_func(n, fw_info)  # Get tensor for the node
        # If we use bias correction, and the node has coefficients to quantize, we need to make sure
        # its previous nodes' tensors are consistent with this node.
        # TODO: factor tensor marking in case of bias correction.
        if qc.weights_bias_correction and fw_info.in_kernel_ops(n):
            for ie in graph.incoming_edges(n):
                input_node = ie.source_node
                create_tensor2node(graph,
                                   input_node)
        if t is not None:
            graph.set_out_stats_collector_to_node(n, t)
