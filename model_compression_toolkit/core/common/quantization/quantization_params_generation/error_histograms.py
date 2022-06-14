# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

def _mse_error_histogram(q_bins: np.ndarray,
                         q_count: np.ndarray,
                         bins: np.ndarray,
                         counts: np.ndarray) -> np.float:
    """
    Compute the error function between a histogram to its quantized version.
    The error is computed based on the mean square error the distributions have.

    Args:
        q_bins: Bins values of the quantized histogram.
        q_count: Bins counts of the quantized histogram.
        bins: Bins values of the original histogram.
        counts: Bins counts of the original histogram.

    Returns:
        MSE between the two histograms.
    """

    return np.sum((np.power((q_bins - bins)[:-1], 2.0) * counts)) / np.sum(counts)


def _mae_error_histogram(q_bins: np.ndarray,
                         q_count: np.ndarray,
                         bins: np.ndarray,
                         counts: np.ndarray) -> np.ndarray:
    """
    Compute the error function between a histogram to its quantized version.
    The error is computed using the mean absolute error between the two histograms.

    Args:
        q_bins: Bins values of the quantized histogram.
        q_count: Bins counts of the quantized histogram.
        bins: Bins values of the original histogram.
        counts: Bins counts of the original histogram.

    Returns:
        Mean absolute error of the two histograms.
    """

    return np.sum((np.abs((q_bins - bins)[:-1]) * counts)) / np.sum(counts)


def _lp_error_histogram(q_bins: np.ndarray,
                        q_count: np.ndarray,
                        bins: np.ndarray,
                        counts: np.ndarray,
                        p: int) -> np.float:
    """
    Compute the error function between a histogram to its quantized version.
    The error is computed based on the distance in Lp-norm between the two distributions.
    The p-norm to use for the distance computing is passed.

    Args:
        q_bins: Bins values of the quantized histogram.
        q_count: Bins counts of the quantized histogram.
        bins: Bins values of the original histogram.
        counts: Bins counts of the original histogram.
        p: p-norm to use for the Lp-norm distance.

    Returns:
        The Lp-norm distance between the two histograms.
    """

    return np.sum((np.power(np.abs((q_bins - bins)[:-1]), p) * counts)) / np.sum(counts)
