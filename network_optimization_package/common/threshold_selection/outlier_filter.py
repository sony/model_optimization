# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


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
