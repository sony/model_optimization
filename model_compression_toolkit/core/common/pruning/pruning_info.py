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


class PruningInfo:
    """
    Class to store metadata about a pruned model, including pruning statistics,
    masks, importance scores.
    """
    def __init__(self,
                 pruning_masks: Dict[BaseNode, np.ndarray],
                 importance_scores: Dict[BaseNode, np.ndarray]):
        """

        Args:
            pruning_masks:
            importance_scores:
        """
        self.pruning_masks = pruning_masks  # Dictionary to store pruning masks for each layer
        self.importance_scores = importance_scores  # Dictionary to store importance scores for each layer

    def get_pruning_mask(self):
        """

        Returns:

        """
        return self.pruning_masks

    def get_importance_score(self):
        """

        Returns:

        """
        return self.importance_scores


