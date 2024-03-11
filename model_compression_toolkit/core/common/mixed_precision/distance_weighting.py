# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from functools import partial

import numpy as np


def get_average_weights(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Get weights for weighting the sensitivity among different layers when evaluating MP configurations on
    model's sensitivity. This function returns equal weights for each layer, such that the sensitivity
    is averaged over all layers.

    Args:
        distance_matrix: Numpy array at shape (L,M): L -number of interest points, M number of samples.
        The matrix contain the distance for each interest point at each sample.

    Returns:
        Numpy array containing equal weights for sensitivity weighting.
    """

    num_nodes = len(distance_matrix)
    return np.asarray([1 / num_nodes for _ in range(num_nodes)])


def get_last_layer_weights(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Get weights for weighting the sensitivity among different layers when evaluating MP configurations on
    model's sensitivity. This function returns weights for each layer, such that the sensitivity
    is computed using only the last layer of the model (all other weights are zero).

    Args:
        distance_matrix: Numpy array at shape (L,M): L -number of interest points, M number of samples.
        The matrix contain the distance for each interest point at each sample.

    Returns:
        Numpy array containing weights for sensitivity weighting (all zero but the last one).
    """
    num_nodes = len(distance_matrix)
    w = np.asarray([0 for _ in range(num_nodes)])
    w[-1] = 1
    return w


class MpDistanceWeighting(Enum):
    """
    Defines mixed precision distance metric weighting methods.
    The enum values can be used to call a function on a set of arguments and key-arguments.

     AVG - take the average distance on all computed layers.

     LAST_LAYER - take only the distance of the last layer output.

    """

    AVG = partial(get_average_weights)
    LAST_LAYER = partial(get_last_layer_weights)

    def __call__(self, distance_matrix: np.ndarray) -> np.ndarray:
        return self.value(distance_matrix)
