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

from typing import List, Tuple

from model_compression_toolkit.core import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import \
    MixedPrecisionSearchManager
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.logger import Logger


def greedy_solution_refinement_procedure(mp_solution: List[int],
                                         search_manager: MixedPrecisionSearchManager,
                                         target_resource_utilization: ResourceUtilization) -> List[int]:
    """
    A greedy procedure to try and improve a mixed-precision solution that was found by a mixed-precision optimization
    algorithm.
    This procedure tries to increase the bit-width precision of configurable nodes that did not get the maximal
    candidate
    in the found solution.
    It iteratively goes over all such nodes, computes the resource utilization values on a modified configuration (with the node's next
    best candidate), filters out all configs that hold the resource utilization constraints and chooses one of them as an improvement
    step
    The choice is done in a greedy approach where we take the configuration that modifies the resource utilization the least.

    Args:
        mp_solution: A mixed-precision configuration that was found by a mixed-precision optimization algorithm.
        search_manager: A MixedPrecisionSearchManager object.
        target_resource_utilization: The target resource utilization for the mixed-precision search.

    Returns: A new, possibly updated, mixed-precision bit-width configuration.

    """
    # Refinement is not supported for BOPs utilization for now...
    if target_resource_utilization.bops_restricted():
        Logger.info(f'Target resource utilization constraint BOPs - Skipping MP greedy solution refinement')
        return mp_solution

    new_solution = mp_solution.copy()
    changed = True

    while changed:
        changed = False
        nodes_ru = {}
        nodes_next_candidate = {}

        for node_idx in range(len(mp_solution)):
            if new_solution[node_idx] == 0:
                # layer has max config in the given solution, nothing to optimize
                continue

            current_node = search_manager.graph.get_configurable_sorted_nodes(search_manager.fw_info)[node_idx]
            node_candidates = current_node.candidates_quantization_cfg

            # only weights kernel attribute is quantized with weights mixed precision
            kernel_attr = search_manager.fw_info.get_kernel_op_attributes(current_node.type)
            kernel_attr = None if kernel_attr is None else kernel_attr[0]
            valid_candidates = _get_valid_candidates_indices(node_candidates,
                                                             new_solution[node_idx],
                                                             target_resource_utilization.activation_restricted(),
                                                             target_resource_utilization.weight_restricted(),
                                                             kernel_attr)

            # Create a list of ru for the valid candidates.
            updated_ru = []
            for valid_idx in valid_candidates:
                node_updated_ru = search_manager.compute_resource_utilization_for_config(
                    config=search_manager.replace_config_in_index(new_solution, node_idx, valid_idx))
                updated_ru.append(node_updated_ru)

            # filter out new configs that don't hold the resource utilization restrictions
            node_filtered_ru = [(node_idx, ru) for node_idx, ru in zip(valid_candidates, updated_ru)
                                if target_resource_utilization.is_satisfied_by(ru)]

            if len(node_filtered_ru) > 0:
                sorted_by_ru = sorted(node_filtered_ru, key=lambda node_ru: (node_ru[1].total_memory,
                                                                             node_ru[1].weights_memory,
                                                                             node_ru[1].activation_memory))
                nodes_ru[node_idx] = sorted_by_ru[0][1]
                nodes_next_candidate[node_idx] = sorted_by_ru[0][0]

        if len(nodes_ru) > 0:
            # filter out new configs that don't hold the ru restrictions
            node_filtered_ru = [(node_idx, ru) for node_idx, ru in nodes_ru.items()]
            sorted_by_ru = sorted(node_filtered_ru, key=lambda node_ru: (node_ru[1].total_memory,
                                                                         node_ru[1].weights_memory,
                                                                         node_ru[1].activation_memory))

            node_idx_to_upgrade = sorted_by_ru[0][0]
            new_solution[node_idx_to_upgrade] = nodes_next_candidate[node_idx_to_upgrade]
            changed = True

    if any([mp_solution[i] != new_solution[i] for i in range(len(mp_solution))]):
        Logger.info(f'Greedy MP algorithm changed configuration from (numbers represent indices of the '
                    f'chosen bit-width candidate for each layer):\n{mp_solution}\nto\n{new_solution}')

    return new_solution


def is_activation_restriction_compliant(
        current_activation_bits: int,
        current_weight_bits: int,
        candidate_activation_bits: int,
        candidate_weight_bits: int,
) -> bool:
    """Used when we try to improve activation bit-width. The comparison of the weight bit width
    is dependent on whether the target RU limits weight usage, thus we look for a MP solution
    for weights, or not."""
    return (candidate_activation_bits > current_activation_bits and
            candidate_weight_bits == current_weight_bits)


def is_weight_restriction_compliant(
        current_activation_bits: int,
        current_weight_bits: int,
        candidate_activation_bits: int,
        candidate_weight_bits: int,
        is_activation_restricted: bool
) -> bool:
    """Used when we try to improve weights bit-width. The comparison of the activation bit width
    is dependent on whether the target RU limits activation usage, thus we look for a MP solution
    for activations, or not."""
    if is_activation_restricted:
        return (candidate_weight_bits > current_weight_bits and
                candidate_activation_bits >= current_activation_bits)
    return (candidate_weight_bits > current_weight_bits and
            candidate_activation_bits == current_activation_bits)


def get_candidate_bits(candidate, kernel_attr: str = None) -> Tuple[int, int]:
    """
    Extract weight and activation bits from a candidate.

    Returns:
        Tuple of (weights_n_bits, activation_n_bits)
    """

    weights_n_bits = candidate.weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits if kernel_attr else None
    activation_n_bits = candidate.activation_quantization_cfg.activation_n_bits
    return weights_n_bits, activation_n_bits


def _get_valid_candidates_indices(
        node_candidates: List['CandidateNodeQuantizationConfig'],
        current_chosen_index: int,
        is_activation_restricted: bool,
        is_weight_restricted: bool,
        kernel_attr: str = None
) -> List[int]:
    """
    Find node's valid candidates to improve the node's MP chosen candidate.

    Valid indices are indices of candidates that have higher number of bits for both
    weights and activations (if they are quantized in this node).

    Args:
        node_candidates: Candidates of the node.
        current_chosen_index: Current index in MP configuration of the node.
        is_activation_restricted: Whether activation quantization is restricted.
        is_weight_restricted: Whether weight quantization is restricted.
        kernel_attr: The name of the kernel attribute on the node, otherwise None.

    Returns:
        List of indices of valid candidates.
    """
    current_candidate = node_candidates[current_chosen_index]
    current_weight_bits, current_activation_bits = get_candidate_bits(current_candidate, kernel_attr)


    def is_valid_candidate(candidate) -> bool:
        candidate_weight_bits, candidate_activation_bits = get_candidate_bits(candidate, kernel_attr)

        if kernel_attr is None:
            # Activation is quantized. Try to increase activation bits for current node.
            return is_activation_restriction_compliant(
                current_activation_bits,
                current_weight_bits,
                candidate_activation_bits,
                candidate_weight_bits
            )
        # Weight is quantized. Try to increase weight bits for current node.
        return is_weight_restriction_compliant(
            current_activation_bits,
            current_weight_bits,
            candidate_activation_bits,
            candidate_weight_bits,
            is_activation_restricted
        )

    return [
        idx for idx, candidate in enumerate(node_candidates)
        if is_valid_candidate(candidate)
    ]
