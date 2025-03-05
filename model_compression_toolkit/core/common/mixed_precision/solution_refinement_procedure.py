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

from typing import List, Tuple, Dict

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

            current_node = search_manager.mp_topo_configurable_nodes[node_idx]
            node_candidates = current_node.candidates_quantization_cfg

            # only weights kernel attribute is quantized with weights mixed precision
            valid_candidates = _get_valid_candidates_indices(node_candidates,
                                                             new_solution[node_idx],
                                                             target_resource_utilization.activation_restricted(),
                                                             target_resource_utilization.weight_restricted()
                                                             )

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


def _get_valid_candidates_indices(
        node_candidates: List[CandidateNodeQuantizationConfig],
        current_chosen_index: int,
        is_activation_restricted: bool,
        is_weight_restricted: bool
) -> List[int]:
    """
    Find node's valid candidates to improve the node's MP chosen candidate.

    Valid indices are indices of candidates that have a higher number of bits for both
    weights and activations. In cases where weights or activations are not restricted (thus,
    we do not search for an MP solution for them), a candidate is considered to be
    valid only if the bit-width of the part that is not eligible for MP has equal bit-width
    to the current candidate.

    Args:
        node_candidates (List[CandidateNodeQuantizationConfig]): List of candidate configurations for the node.
        current_chosen_index (int): Index of the currently chosen candidate in the list.
        is_activation_restricted (bool): Indicates whether activation resources are restricted.
        is_weight_restricted (bool): Indicates whether weight resources are restricted.


    Returns:
        List[int]: List of indices of valid candidates.
    """

    def get_candidate_bits(candidate: CandidateNodeQuantizationConfig) -> Tuple[Dict[str, int], int]:
        """
        Extract weight and activation bits from a candidate.

        Args:
            candidate (CandidateNodeQuantizationConfig): A candidate node configuration.

        Returns:
            Tuple[Dict[str, int], int]:
                - A dictionary mapping weight attributes to their bit-widths.
                - The activation bit-width.
        """
        weight_attrs = candidate.weights_quantization_cfg.all_weight_attrs
        weight_attrs_to_nbits: Dict[str, int] = {
            w_attr: candidate.weights_quantization_cfg.get_attr_config(w_attr).weights_n_bits
            for w_attr in weight_attrs
        }
        activation_n_bits: int = candidate.activation_quantization_cfg.activation_n_bits
        return weight_attrs_to_nbits, activation_n_bits

    def is_valid_candidate(candidate: CandidateNodeQuantizationConfig) -> bool:
        """
        Check if a candidate satisfies the weight and activation bit-width constraints to be
        considered as a valid candidate.

        Args:
            candidate (CandidateNodeQuantizationConfig): A candidate node configuration.

        Returns:
            bool: True if the candidate is valid, False otherwise.
        """
        candidate_weight_bits, candidate_activation_bits = get_candidate_bits(candidate)

        # Current candidate - no need to check
        if candidate_activation_bits == current_activation_bits and all(candidate_weight_bits[attr] == current_weight_bits[attr] for attr in current_weight_bits.keys()):
            return False

        valid_weight = all(
            candidate_weight_bits[attr] >= current_weight_bits[attr] if is_weight_restricted else
            candidate_weight_bits[attr] == current_weight_bits[attr]
            for attr in current_weight_bits.keys()
        )
        valid_activation = (
            candidate_activation_bits >= current_activation_bits if is_activation_restricted else
            candidate_activation_bits == current_activation_bits
        )
        return valid_weight and valid_activation

    current_candidate = node_candidates[current_chosen_index]
    current_weight_bits, current_activation_bits = get_candidate_bits(current_candidate)

    return [idx for idx, candidate in enumerate(node_candidates) if is_valid_candidate(candidate)]