# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig

from model_compression_toolkit.core.common.quantization.quantization_params_fn_selection import \
    get_activation_quantization_params_fn
from model_compression_toolkit.target_platform_capabilities import OpQuantizationConfig
from model_compression_toolkit.core.common.framework_info import FrameworkInfo


def filter_nodes_candidates(graph: Graph):
    """
    Filters the graph's nodes candidates configuration list.
    We apply this after mark activation operation to eliminate nodes that their activation are no longer being quantized
    from the mixed-precision search.
    Updating the lists is preformed inplace on the graph object.

    Args:
        graph: Graph for which to add quantization info to each node.
    """
    nodes = list(graph.nodes)
    fusing_info = graph.fusing_info
    for n in nodes:
        fused_node_op_id = fusing_info.get_fused_op_id_for_node(n.name)
        fusing_op_quantization_cfg = fusing_info.get_fused_op_quantization_config(fused_node_op_id)

        n.candidates_quantization_cfg = filter_node_candidates(node=n, fw_info=graph.fw_info, op_cfg=fusing_op_quantization_cfg)

    return graph


def _filter_bit_method_dups(candidates: List[CandidateNodeQuantizationConfig],
                            kernel_attr: str = None) -> List[CandidateNodeQuantizationConfig]:
    """
    Filters out duplications in candidates configuration list, based on similarity in
    (weights_n_bits, weights_quantization_method, activation_n_bits, activation_quantization_method).
    Weights quantization configuration considers only kernel attributes.

    Args:
        candidates: A list of quantization configuration candidates.
        kernel_attr: The name of the node's kernel attribute if such exists.

    Returns: A filtered list of quantization configuration candidates.

    """
    seen_bits_method_combinations = set()
    final_candidates = []
    for c in candidates:
        weight_n_bits = None if kernel_attr is None else (
            c.weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits)
        weights_quantization_method = None if kernel_attr is None else (
            c.weights_quantization_cfg.get_attr_config(kernel_attr).weights_quantization_method)
        comb = (weight_n_bits,
                weights_quantization_method,
                c.activation_quantization_cfg.activation_n_bits,
                c.activation_quantization_cfg.activation_quantization_method)
        if comb not in seen_bits_method_combinations:
            final_candidates.append(c)
            seen_bits_method_combinations.add(comb)

    return final_candidates


def filter_node_candidates(node: BaseNode, fw_info: FrameworkInfo, op_cfg: OpQuantizationConfig) -> List[CandidateNodeQuantizationConfig]:
    """
    Updates a node's candidates configuration list.
    If the node's weights quantization is disabled (or it only has activations to quantize), then the updated list
    will have a candidate with any of the different original activation bitwidths candidates and a default value
    for its weights bitwidth (that doesn't have any impact on the quantization or the mixed-precision search.
    If the node's activation quantization is disabled, the same filtering applied for the weights bitwidth candidates.

    Args:
        node: Node to set its quantization configurations.
        fw_info: FrameworkInfo object with information about the specific framework's model.
        op_cfg:OpQuantizationConfig to set.
    """

    filtered_candidates = copy.deepcopy(node.candidates_quantization_cfg)
    final_candidates = copy.deepcopy(node.candidates_quantization_cfg)
    kernel_attr = fw_info.get_kernel_op_attributes(node.type)[0]

    if (kernel_attr is None or not node.is_weights_quantization_enabled(kernel_attr)) and not node.is_activation_quantization_enabled():
        # If activation quantization is disabled and the node doesn't have a kernel or doesn't quantize the kernel,
        # but for some reason the node has multiple candidates then replace it with a single dummy candidate with
        # default bit-width values.
        single_dummy_candidate = filtered_candidates[0]
        _update_activation_quantization_cfg(node, single_dummy_candidate, op_cfg)

        if kernel_attr is not None:
            kernel_config = single_dummy_candidate.weights_quantization_cfg.get_attr_config(kernel_attr)
            kernel_config.weights_n_bits = FLOAT_BITWIDTH
            kernel_config.weights_quantization_method = QuantizationMethod.POWER_OF_TWO

        final_candidates = [single_dummy_candidate]

    elif not node.is_activation_quantization_enabled():
        # Remove candidates that have duplicated weights candidates for node with disabled activation quantization.
        # Replacing the activation n_bits in the remained configurations with default value to prevent confusion.
        seen_candidates = set()
        filtered_candidates = [candidate for candidate in filtered_candidates if
                               candidate.weights_quantization_cfg not in seen_candidates
                               and not seen_candidates.add(candidate.weights_quantization_cfg)]

        for c in filtered_candidates:
            _update_activation_quantization_cfg(node, c, op_cfg)

        final_candidates = _filter_bit_method_dups(filtered_candidates, kernel_attr)

    elif kernel_attr is None or not node.is_weights_quantization_enabled(kernel_attr):
        # TODO:
        #  To allow MP on positional weights we need to modify this to consider all weights not only kernel.
        # Remove candidates that have duplicated activation candidates for node with disabled weights quantization.
        # Replacing the weights n_bits in the remained configurations with default value to prevent confusion.
        seen_candidates = set()
        filtered_candidates = [candidate for candidate in filtered_candidates if
                               candidate.activation_quantization_cfg not in seen_candidates
                               and not seen_candidates.add(candidate.activation_quantization_cfg)]

        for c in filtered_candidates:
            if kernel_attr is not None:
                kernel_config = c.weights_quantization_cfg.get_attr_config(kernel_attr)
                kernel_config.weights_n_bits = FLOAT_BITWIDTH
                kernel_config.weights_quantization_method = QuantizationMethod.POWER_OF_TWO

        final_candidates = _filter_bit_method_dups(filtered_candidates, kernel_attr)

    return final_candidates


def _update_activation_quantization_cfg(node: BaseNode,
                                        candidate: CandidateNodeQuantizationConfig,
                                        op_cfg: OpQuantizationConfig = None):
    """
    Updates the activation quantization configuration of a candidate node.
    If the node is FLN quantization, it will use the op_cfg values.
    Otherwise, it will set the activation n_bits to FLOAT_BITWIDTH and the quantization method to POWER_OF_TWO.
    Args:
        node: Node to set its quantization configurations.
        candidate: CandidateNodeQuantizationConfig to update.
        op_cfg: OpQuantizationConfig of the fln node with quantizers types to use when creating fln node quantization configuration.
    """
    actq_cfg = candidate.activation_quantization_cfg

    if node.is_fln_quantization() and op_cfg is not None:
        activation_n_bits = op_cfg.activation_n_bits
        signedness = op_cfg.signedness
        activation_quantization_method = op_cfg.activation_quantization_method
    else:
        activation_n_bits = FLOAT_BITWIDTH
        signedness = actq_cfg.signedness
        activation_quantization_method = QuantizationMethod.POWER_OF_TWO

    activation_quantization_fn = get_activation_quantization_params_fn(activation_quantization_method)

    actq_cfg.activation_n_bits = activation_n_bits
    actq_cfg.signedness = signedness
    actq_cfg.activation_quantization_method = activation_quantization_method
    actq_cfg.activation_quantization_fn = activation_quantization_fn
    actq_cfg.activation_quantization_params_fn = activation_quantization_fn