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

from typing import Any, List

from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common import Graph, BaseNode
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.logger import Logger
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig



def set_bit_widths(quant_config: QuantizationConfig,
                   graph_to_set_bit_widths: Graph,
                   fw_info: FrameworkInfo = None,
                   bit_widths_config: List[int] = None) -> Graph:
    """
    Set bit widths configuration to nodes in a graph. For each node, use the desired index
    in bit_widths_config to finalize the node weights quantization configuration.

    Args:
        quant_config: MixedPrecisionQuantizationConfig the graph was computed according to.
        graph_to_set_bit_widths: A prepared for quantization graph to set its bit widths.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        bit_widths_config: MP configuration (a list of indices: one for each node's candidate quantization configuration).

    """
    graph = copy.deepcopy(graph_to_set_bit_widths)

    if isinstance(quant_config, MixedPrecisionQuantizationConfig):
        if len(quant_config.weights_n_bits) == 0:
            Logger.critical(
                f'Quantization configuration nbits has to contain at least one bit width. Length is: '
                f'{len(quant_config.weights_n_bits)}')

        # When working in mixed-precision mode, and there's only one bitwidth, we simply set the
        # only candidate of the node as its final weight quantization configuration.
        if len(quant_config.weights_n_bits) == 1:
            for n in graph.nodes:
                if n.name in graph.get_configurable_sorted_nodes_names():
                    assert len(n.candidates_weights_quantization_cfg) == 1
                    n.final_weights_quantization_cfg = n.candidates_weights_quantization_cfg[0]

        else:
            Logger.info(f'Set bit widths from configuration: {bit_widths_config}')
            # Get a list of nodes' names we need to finalize (that they have at least one weight qc candidate).
            sorted_nodes_names = graph.get_configurable_sorted_nodes_names()
            for node in graph.nodes:  # set a specific node qc for each node final weights qc
                node_name = node.name if not node.reuse else '_'.join(node.name.split('_')[:-2]) # if it's reused, take the configuration that the base node has
                if node_name in sorted_nodes_names:  # only configurable nodes are in this list
                    node_index_in_graph = sorted_nodes_names.index(node_name)
                    _set_node_qc(bit_widths_config,
                                 fw_info,
                                 node,
                                 node_index_in_graph)

    # When working in non-mixed-precision mode, there's only one bitwidth, and we simply set the
    # only candidate of the node as its final weight quantization configuration.
    else:
        for n in graph.nodes:
            if fw_info.in_kernel_ops(n):
                assert len(n.candidates_weights_quantization_cfg) == 1
                n.final_weights_quantization_cfg = n.candidates_weights_quantization_cfg[0]

    return graph


def _get_node_qc_by_bit_widths(node: BaseNode,
                               bit_width_cfg: List[int],
                               node_index_in_graph: int) -> Any:
    """
    Get the node's quantization configuration that
    matches to the bit width index as in the MP configuration bit_width_cfg.
    If it was not found, return None.

    Args:
        node: Node to get its quantization configuration candidate.
        bit_width_cfg: Configuration which determines the node's desired bit width.
        node_index_in_graph: Index of the node in the bit_width_cfg.

    Returns:
        Node quantization configuration if it was found, or None otherwise.
    """

    if node.candidates_weights_quantization_cfg is not None:
        bit_index_in_cfg = bit_width_cfg[node_index_in_graph]
        qc = node.candidates_weights_quantization_cfg[bit_index_in_cfg]
        return qc
    return None


def _set_node_qc(bit_width_cfg: List[int],
                 fw_info: FrameworkInfo,
                 node: BaseNode,
                 node_index_in_graph: int):
    """
    Get the node's quantization configuration that
    matches to the bit width index as in the MP configuration bit_width_cfg, and use it to finalize the node's
    weights quantization config.
    If the node quantization config was not found, raise an exception.

    Args:
        bit_width_cfg: Configuration which determines the node's desired bit width.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        node: Node to set its node quantization configuration.
        node_index_in_graph: Index of the node in the bit_width_cfg.

    """
    if fw_info.in_kernel_ops(node):
        node_qc = _get_node_qc_by_bit_widths(node, bit_width_cfg, node_index_in_graph)
        if node_qc is None:
            Logger.critical(f'Node {node.name} quantization configuration from configuration file'
                            f' was not found in candidates configurations.')
        else:
            node.final_weights_quantization_cfg = node_qc
