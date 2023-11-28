from typing import Dict, List
import numpy as np

class PruningInfo:
    """
    Class to store metadata about a pruned model, including pruning statistics,
    masks, importance scores, and parameter counts.
    """
    def __init__(self,
                 pruning_masks,
                 importance_scores):
        self.pruning_masks = pruning_masks  # Dictionary to store pruning masks for each layer
        self.importance_scores = importance_scores  # Dictionary to store importance scores for each layer

    def get_pruning_mask(self):
        return self.pruning_masks

    def get_importance_score(self) -> np.ndarray:
        return self.importance_scores


