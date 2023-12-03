from enum import Enum

from model_compression_toolkit.constants import PRUNING_NUM_SCORE_APPROXIMATIONS


class ImportanceMetric(Enum):
    """
    Enum for specifying the metric used to determine the importance of channels when pruning.
    """
    LFH = 0  # Hessian approximation based on weights, to determine channel importance without explicit labels.


class ChannelsFilteringStrategy(Enum):
    """
    Enum for specifying the strategy used for filtering (pruning) channels.
    """
    GREEDY = 0  # Greedy strategy for pruning channels based on importance metrics.


class PruningConfig:
    """
    Configuration class for specifying how a neural network should be pruned.

    Attributes:
        num_score_approximations (int): The number of score approximations to perform
                                        when calculating channel importance.
        importance_metric (ImportanceMetric): The metric used to calculate channel importance.
        channels_filtering_strategy (ChannelsFilteringStrategy): The strategy used to filter out channels.
    """

    def __init__(self,
                 num_score_approximations: int = PRUNING_NUM_SCORE_APPROXIMATIONS,
                 importance_metric: ImportanceMetric = ImportanceMetric.LFH,
                 channels_filtering_strategy: ChannelsFilteringStrategy = ChannelsFilteringStrategy.GREEDY):
        """
        Initializes a PruningConfig object with default or specified parameters.

        Args:
            num_score_approximations (int): The number of times to approximate the scoring
                                            for channel importance. Defaults to a predefined
                                            constant value.
            importance_metric (ImportanceMetric): The method used for calculating the importance
                                                  of channels in a network. Defaults to label-free
                                                  Hessian (LFH) approximation.
            channels_filtering_strategy (ChannelsFilteringStrategy): The strategy for selecting
                                                                     which channels to prune.
                                                                     Defaults to a greedy approach.
        """

        # The number of times the importance score is approximated.
        self.num_score_approximations = num_score_approximations

        # The metric used to assess the importance of each channel in a layer.
        self.importance_metric = importance_metric

        # The strategy to use when deciding which channels to prune based on their importance scores.
        self.channels_filtering_strategy = channels_filtering_strategy

        # TODO: Consider limiting ratio per layer
