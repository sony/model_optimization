# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit.core.common.collectors.histogram_collector import HistogramCollector


class WeightedHistogramCollector(HistogramCollector):
    """
    Collector for holding weighted histograms of tensors going through it.
    Extends the functionality of the base HistogramCollector by incorporating weights
    into the histogram calculation, allowing for weighted distributions.
    """
    def __init__(self, n_bins: int = 2048):
        """
        Args:
            n_bins: Number of bins in the histogram.
        """

        super().__init__(n_bins)

    def update(self, x: np.ndarray, weights: np.ndarray = None):
        """
        Update the current state of the histogram bins and counts based on a new
        tensor that passes through the collector, taking weights into account.

        Args:
            x: Tensor that passes through the collector to update the histogram.
            weights: Array of weights corresponding to the elements of the tensor `x`.
                     If not provided, uniform weights of 1 will be applied to all elements.

        Details:
            - The `weights` parameter allows each element of `x` to contribute a
              weighted amount to the histogram bins, rather than contributing a simple count.
            - This is particularly useful when the data being processed has an associated
              importance, frequency, or probability value that should influence the histogram.
              For example:
                - Hessian values can serve as weights for each activation in `x`.
                - Normalization of the histogram to reflect relative contributions.
            - The method ensures that `x` and `weights` have matching shapes and logs
              an error if this condition is not met.
        """
        if weights is None or weights.sum() == 0:
            weights = np.ones_like(x) # Assign uniform weights if none are provided.

        if x.shape != weights.shape:
            # Get x's shape
            x_shape = x.shape

            # Get weight's shape
            weights_shape = weights.shape

            # Determine the correct shape for weights
            weights_new_shape = list(weights_shape)  # Convert to list for modification

            # Ensure weights has the same number of dimensions as x
            while len(weights_new_shape) < len(x_shape):
                weights_new_shape.append(1)  # Add singleton dimensions

            # Reshape weights to the correct shape
            weights = weights.reshape(weights_new_shape)

            # Broadcast weights to match x's shape
            weights = np.broadcast_to(weights, x_shape)

        # Compute the weighted histogram.
        count, bins = np.histogram(x, bins=self._n_bins, weights=weights)

        # Store the weighted histogram (counts and bins) for this iteration.
        self._histogram_per_iteration.append((count, bins))
