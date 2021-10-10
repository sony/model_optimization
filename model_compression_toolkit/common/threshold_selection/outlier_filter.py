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


def z_score_filter(z_threshold: float,
                   bins: np.ndarray,
                   counts: np.ndarray):
    """
    Filter outliers in a histogram using z-score for outliers detection.
    For more info: https://www.geeksforgeeks.org/z-score-for-outlier-detection-python/

    Args:
        z_threshold: Threshold to remove outliers with z-score above it.
        bins: Numpy array with bins ranges of the histogram.
        counts: Numpy array with elements count in ranges of the histogram.

    Returns:
        Numpy array with elements count in ranges of the histogram after z-score filtering.
    """

    bins = np.copy(bins)
    counts = np.copy(counts)
    bins = bins[:-1]  # take out the last range

    # Compute the z-score
    mu = np.sum(bins * counts) / np.sum(counts)
    sigma = np.sqrt(np.sum(np.power(bins - mu, 2.0) * counts) / np.sum(counts))
    z_score = np.abs(bins - mu) / sigma

    index2zero = z_score > z_threshold  # indices to zero as they are probably outliers
    counts[index2zero] = 0

    return counts
