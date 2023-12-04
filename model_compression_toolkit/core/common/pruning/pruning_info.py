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


