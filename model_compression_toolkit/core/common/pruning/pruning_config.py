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

from enum import Enum

from model_compression_toolkit.constants import PRUNING_NUM_SCORE_APPROXIMATIONS


class ImportanceMetric(Enum):
    """
    Enum for specifying the metric used to determine the importance of channels when pruning:

    LFH - Label-Free Hessian uses hessian info for measuring each channel's sensitivity.

    """
    LFH = 0  # Score based on the Hessian matrix w.r.t. layers weights, to determine channel importance without labels.


class ChannelsFilteringStrategy(Enum):
    """
    Enum for specifying the strategy used for filtering (pruning) channels:

    GREEDY - Prune the least important channel groups up to the allowed resources utilization limit (for now, only weights_memory is considered).

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

