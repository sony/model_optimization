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
from model_compression_toolkit.logger import Logger


def check_broadcastable(x: np.ndarray, weights: np.ndarray) -> None:
    """
    Checks if tensor 'weights' can be broadcasted to the shape of tensor 'x'.

    Args:
        x (np.ndarray): The target tensor.
        weights (np.ndarray): The tensor to check broadcasting compatibility.

    Raises:
        Logger.critical: If 'weights' cannot be broadcasted to 'x'.
    """
    # Get shapes
    shape_x = x.shape
    shape_w = weights.shape

    # Ensure weights has less or equal dimensions than x
    if len(shape_w) > len(shape_x):
        Logger.critical(f"Tensor weights with shape {shape_w} has more dimensions than tensor a with shape {shape_x}.")

    # Align shapes by padding weights' shape with leading ones
    shape_w = (1,) * (len(shape_x) - len(shape_w)) + shape_w

    # Check if each dimension is either equal or 1
    for i, (sx, sw) in enumerate(zip(shape_x, shape_w)):
        if not (sx == sw or sw == 1):
            Logger.critical(f"Tensor weights with shape {shape_w} cannot be broadcasted to tensor a with shape {shape_x}. "
                             f"Dimension mismatch at index {i}: {sw} cannot be broadcasted to {sx}.")


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
        if weights is None or np.all(weights == 0):
            weights = np.ones_like(x) # Assign uniform weights if none are provided.

        # Checks if tensor 'weights' can be broadcasted to the shape of tensor 'x'.
        check_broadcastable(x, weights)

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
