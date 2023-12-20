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

from typing import List, Dict, Tuple

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode
import numpy as np


class ChannelGrouping:
    """
    ChannelGrouping handles the sorting and grouping of channel indices for prunable nodes in a graph,
    based on their importance scores and SIMD group sizes.
    """

    def __init__(self,
                 prunable_nodes: List[BaseNode],
                 fw_info: FrameworkInfo):
        """
        Initializes the ChannelGrouping with necessary information.

        Args:
            prunable_nodes: List of nodes that can be pruned.
            fw_info: Framework-specific information and utilities.
        """
        self.prunable_nodes = prunable_nodes
        self.fw_info = fw_info
        # Store for each node a list of numpy arrays. Each numpy array represents the
        # indices of the channels in an SIMD group.
        self.simd_groups_indices = {}

    def group_scores_by_simd_groups(self, score_by_node: Dict[BaseNode, np.ndarray]):
        for prunable_node, node_scores in score_by_node.items():
            self.simd_groups_indices[prunable_node] = self._group_scores_by_simd_size(node_scores,
                                                                                      prunable_node.get_simd())

    def _group_scores_by_simd_size(self,
                                   scores: np.ndarray,
                                   simd: int) -> List[np.ndarray]:
        """
        Groups the scores and their corresponding indices based on SIMD size.

        Args:
            scores: An array of scores to be grouped.
            simd: Size of the SIMD group.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Grouped scores and indices.
        """
        sorted_indices = np.argsort(-scores)
        num_complete_groups = len(scores) // simd
        scores_groups = [scores[sorted_indices[i * simd:(i + 1) * simd]] for i in range(num_complete_groups)]
        indices_groups = [sorted_indices[i * simd:(i + 1) * simd] for i in range(num_complete_groups)]
        remainder = len(scores) % simd
        if remainder != 0:
            scores_groups.append(scores[sorted_indices[-remainder:]])
            indices_groups.append(sorted_indices[-remainder:])
        return indices_groups

    def get_group_indices(self) -> Dict[BaseNode, List[np.ndarray]]:
        """
        Returns the grouped indices for each prunable node.

        Returns:
            Dict[BaseNode, List[np.ndarray]]: Grouped indices for each node.
        """
        return self.simd_groups_indices
