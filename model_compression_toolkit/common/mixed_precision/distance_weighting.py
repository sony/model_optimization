# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

import numpy as np

from model_compression_toolkit.common import Graph

from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionMetricsWeighting


def _get_average_weights(graph: Graph) -> np.ndarray:
    """
    Get weights for weighting the sensitivity among different layers when evaluating MP configurations on
    model's sensitivity. This function returns equal weights for each layer, such that the sensitivity
    is averaged over all layers.
    Args:
        graph: Graph to compute the weights for its sensitivity evaluation.

    Returns:
        Numpy array containing equal weights for sensitivity weighting.
    """

    num_nodes = len(graph.get_configurable_sorted_nodes())
    return np.asarray([1/num_nodes for _ in range(num_nodes)])


def _get_last_layer_weights(graph: Graph) -> np.ndarray:
    """
    Get weights for weighting the sensitivity among different layers when evaluating MP configurations on
    model's sensitivity. This function returns weights for each layer, such that the sensitivity
    is computed using only the last layer of the model (all other weights are zero).

    Args:
        graph: Graph to compute the weights for its sensitivity evaluation.

    Returns:
        Numpy array containing weights for sensitivity weighting (all zero but the last one).
    """
    num_nodes = len(graph.get_configurable_sorted_nodes())
    w = np.asarray([0 for _ in range(num_nodes)])
    w[-1] = 1
    return w


# Factory for weighting functions.
metric_weighting_dict = {MixedPrecisionMetricsWeighting.AVERAGE: _get_average_weights,
                         MixedPrecisionMetricsWeighting.LAST_LAYER: _get_last_layer_weights}

