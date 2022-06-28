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

from typing import Callable

from networkx.algorithms.dag import topological_sort

from model_compression_toolkit.core import common


def create_tensor2node(graph: common.Graph,
                       node: common.BaseNode,
                       fw_info: common.FrameworkInfo):
    """
    Force tensor creation and assignment for a node.
    Args:
        graph: Graph of the node (for retrieving the current tensor).
        node: Node to create a tensor for.
        fw_info: Specific framework information (for example, output channels index).

    """
    current_tensor = graph.get_out_stats_collector(node)
    is_list_nostat_collectors = isinstance(current_tensor, list) and len([sc for sc in current_tensor if not isinstance(sc, common.NoStatsCollector)]) == 0
    if isinstance(current_tensor, common.NoStatsCollector) or current_tensor is None or is_list_nostat_collectors:
        out_channel_axis = fw_info.out_channel_axis_mapping.get(node.type)
        graph.set_out_stats_collector_to_node(node, common.StatsCollector(out_channel_axis))


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
        sc = node_analyze_func(n, fw_info=fw_info)  # Get tensor for the node
        # If we use bias correction, and the node has coefficients to quantize, we need to make sure
        # its previous nodes' tensors are consistent with this node.
        # TODO: factor tensor marking in case of bias correction.
        if qc.weights_bias_correction and n.is_weights_quantization_enabled():
            for ie in graph.incoming_edges(n):
                input_node = ie.source_node
                create_tensor2node(graph,
                                   input_node,
                                   fw_info)
        if sc is not None:
            graph.set_out_stats_collector_to_node(n, sc)
