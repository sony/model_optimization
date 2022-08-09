# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.logger import Logger


def set_bit_widths(mixed_precision_enable: bool,
                   graph_to_set_bit_widths: Graph,
                   bit_widths_config: List[int] = None) -> Graph:
    """
    Set bit widths configuration to nodes in a graph. For each node, use the desired index
    in bit_widths_config to finalize the node weights and activation quantization configuration.

    Args:
        mixed_precision_enable: Is mixed precision enabled.
        graph_to_set_bit_widths: A prepared for quantization graph to set its bit widths.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel
        channels indices, groups of layers by how they should be quantized, etc.)
        bit_widths_config: MP configuration (a list of indices: one for each node's candidate
        quantization configuration).

    """
    graph = copy.deepcopy(graph_to_set_bit_widths)

    if mixed_precision_enable:
        assert all([len(n.candidates_quantization_cfg) > 0 for n in graph.get_configurable_sorted_nodes()]), \
            "All configurable nodes in graph should have at least one candidate configuration in mixed precision mode"

        Logger.info(f'Set bit widths from configuration: {bit_widths_config}')
        # Get a list of nodes' names we need to finalize (that they have at least one weight qc candidate).
        sorted_nodes_names = graph.get_configurable_sorted_nodes_names()
        for node in graph.nodes:  # set a specific node qc for each node final weights qc
            # If it's reused, take the configuration that the base node has
            node_name = node.name if not node.reuse else '_'.join(node.name.split('_')[:-2])
            if node_name in sorted_nodes_names:  # only configurable nodes are in this list
                node_index_in_graph = sorted_nodes_names.index(node_name)
                _set_node_final_qc(bit_widths_config,
                                   node,
                                   node_index_in_graph)
            elif node.is_activation_quantization_enabled():
                # If we are here, this means that we are in weights-only mixed-precision
                # (i.e., activations are quantized with fixed bitwidth or not quantized)
                # and that this node doesn't have weights to quantize
                assert len(node.candidates_quantization_cfg) > 0, \
                    "Node need to have at least one quantization configuration in order to quantize its activation"
                node.final_activation_quantization_cfg = copy.deepcopy(node.candidates_quantization_cfg[0].activation_quantization_cfg)
            elif node.is_weights_quantization_enabled():
                # If we are here, this means that we are in activation-only mixed-precision
                # (i.e., weights are quantized with fixed bitwidth or not quantized)
                # and that this node doesn't have activations to quantize
                assert len(node.candidates_quantization_cfg) > 0, \
                    "Node need to have at least one quantization configuration in order to quantize its activation"
                node.final_weights_quantization_cfg = copy.deepcopy(node.candidates_quantization_cfg[0].weights_quantization_cfg)

    # When working in non-mixed-precision mode, there's only one bitwidth, and we simply set the
    # only candidate of the node as its final weight and activation quantization configuration.
    else:
        for n in graph.nodes:
            assert len(n.candidates_quantization_cfg) == 1
            n.final_weights_quantization_cfg = copy.deepcopy(n.candidates_quantization_cfg[0].weights_quantization_cfg)
            n.final_activation_quantization_cfg = copy.deepcopy(n.candidates_quantization_cfg[0].activation_quantization_cfg)

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

    if node.is_weights_quantization_enabled() or node.is_activation_quantization_enabled():
        bit_index_in_cfg = bit_width_cfg[node_index_in_graph]
        qc = node.candidates_quantization_cfg[bit_index_in_cfg]
        return qc
    return None


def _set_node_final_qc(bit_width_cfg: List[int],
                       node: BaseNode,
                       node_index_in_graph: int):
    """
    Get the node's quantization configuration that
    matches to the bit width index as in the MP configuration bit_width_cfg, and use it to finalize the node's
    weights and activation quantization config.
    If the node quantization config was not found, raise an exception.

    Args:
        bit_width_cfg: Configuration which determines the node's desired bit width.
        node: Node to set its node quantization configuration.
        node_index_in_graph: Index of the node in the bit_width_cfg.

    """
    node_qc = _get_node_qc_by_bit_widths(node,
                                         bit_width_cfg,
                                         node_index_in_graph)

    if node_qc is None:
        Logger.critical(f'Node {node.name} quantization configuration from configuration file'
                        f' was not found in candidates configurations.')

    else:
        node.final_weights_quantization_cfg = node_qc.weights_quantization_cfg
        node.final_activation_quantization_cfg = node_qc.activation_quantization_cfg
