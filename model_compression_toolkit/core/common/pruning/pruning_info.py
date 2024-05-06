# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Dict, List
import numpy as np

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.logger import Logger


class PruningInfo:
    """
    PruningInfo stores information about a pruned model, including the pruning masks
    and importance scores for each layer. This class acts as a container for accessing
    pruning-related metadata.

    """

    def __init__(self,
                 pruning_masks: Dict[BaseNode, np.ndarray],
                 importance_scores: Dict[BaseNode, np.ndarray]):
        """
        Args:
            pruning_masks (Dict[BaseNode, np.ndarray]): Stores the pruning masks for each layer. A pruning mask is an array where each element indicates whether the corresponding channel or neuron has been pruned (0) or kept (1).
            importance_scores (Dict[BaseNode, np.ndarray]): Stores the importance scores for each layer. Importance scores quantify the significance of each channel in the layer.

        """
        self._pruning_masks = pruning_masks
        self._importance_scores = importance_scores

    @property
    def pruning_masks(self) -> Dict[BaseNode, np.ndarray]:
        """
        The pruning masks for each layer.

        Returns:
            Dict[BaseNode, np.ndarray]: The pruning masks.
        """
        return self._pruning_masks

    @property
    def importance_scores(self) -> Dict[BaseNode, np.ndarray]:
        """
        The importance scores for each layer.

        Returns:
            Dict[BaseNode, np.ndarray]: The importance scores.
        """
        return self._importance_scores

def unroll_simd_scores_to_per_channel_scores(simd_scores: Dict[BaseNode, np.ndarray],
                                             simd_groups_indices: Dict[BaseNode, List[np.ndarray]]) -> Dict[BaseNode, np.ndarray]:
    """
    Expands SIMD group scores into per-channel scores. This is necessary when channels
    are grouped in SIMD groups, and a single score is assigned to each group. The function
    duplicates the group score to each channel in that group.

    Args:
        simd_scores (Dict[BaseNode, np.ndarray]): The scores assigned to each SIMD group.
        simd_groups_indices (Dict[BaseNode, List[np.ndarray]]): The indices of channels in each SIMD group.

    Returns:
        Dict[BaseNode, np.ndarray]: Expanded scores for each individual channel.
    """
    if simd_scores is None or simd_groups_indices is None:
        Logger.critical(f"Failed to find scores and indices to create unrolled scores for pruning information."
                        f" Scores: {simd_scores}, Group indices: {simd_groups_indices}.")  # pragma: no cover
    _scores = {}
    for node, groups_indices in simd_groups_indices.items():
        node_scores = simd_scores[node]
        total_indices = sum(len(group) for group in groups_indices)
        new_node_scores = np.zeros(total_indices)

        for group_score, group_indices in zip(node_scores, groups_indices):
            new_node_scores[group_indices] = group_score

        _scores[node] = new_node_scores
    return _scores
