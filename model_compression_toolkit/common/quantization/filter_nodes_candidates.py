# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from model_compression_toolkit.common import Graph, BaseNode
from model_compression_toolkit.common.constants import DEFAULT_CANDIDATE_BITWIDTH


def filter_nodes_candidates(graph: Graph):
    """
    TODO: documentation
    Add quantization configuration for each graph node.

    Args:
        graph: Graph for which to add quantization info to each node.

    Returns:
        The graph with quantization configurations attached to each node in it.
    """

    configurable_sorted_nodes = graph.get_configurable_sorted_nodes()
    for n in configurable_sorted_nodes:
        n.candidates_quantization_cfg = filter_node_candidates(node=n)


def filter_node_candidates(node: BaseNode):
    """
    # TODO: documentation
    Create and set quantization configurations to a node (for both weights and activation).

    Args:
        node: Node to set its quantization configurations.
    """

    filtered_candidates = copy.deepcopy(node.candidates_quantization_cfg)

    if not node.is_activation_quantization_enabled():
        # Remove candidates that have duplicated weights bitwidth for node with disabled activation quantization.
        # Replacing the activation n_bits in the remained configurations with default value to prevent confusion.
        seen_bitwidth = set()
        filtered_candidates = [candidate for candidate in filtered_candidates if
                               candidate.weights_quantization_cfg.weights_n_bits not in seen_bitwidth
                               and not seen_bitwidth.add(candidate.weights_quantization_cfg.weights_n_bits)]
        for c in filtered_candidates:
            c.activation_quantization_cfg.activation_n_bits = DEFAULT_CANDIDATE_BITWIDTH

    elif not node.is_weights_quantization_enabled():
        # Remove candidates that have duplicated activation bitwidth for node with disabled weights quantization.
        # Replacing the weights n_bits in the remained configurations with default value to prevent confusion.
        seen_bitwidth = set()
        filtered_candidates = [candidate for candidate in filtered_candidates if
                               candidate.activation_quantization_cfg.activation_n_bits not in seen_bitwidth
                               and not seen_bitwidth.add(candidate.activation_quantization_cfg.activation_n_bits)]
        for c in filtered_candidates:
            c.weights_quantization_cfg.weights_n_bits = DEFAULT_CANDIDATE_BITWIDTH

    return filtered_candidates
