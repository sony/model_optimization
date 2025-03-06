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

from typing import Tuple
import numpy as np
from model_compression_toolkit.core.common.collectors.base_collector import BaseCollector


def interpolate_histogram(current_bins: np.ndarray,
                          bins_to_interpolate: np.ndarray,
                          counts_to_interpolate: np.ndarray) -> np.ndarray:
    """
    Interpolate a histogram to new bins values. Return the counts of the histogram as if it
    was collected in current_bins ranges (approximately).
    Args:
        current_bins: Bins to use for interpolation.
        bins_to_interpolate: Bins to interpolate.
        counts_to_interpolate: Counts of the histogram to interpolate.

    Returns:
        Counts of the histogram if it was collected between current_bins values.
    """
    cumulative_hist = np.hstack([0, np.cumsum(counts_to_interpolate)])  # Compute the accumulated histogram
    # Compute the interpolation of the accumulated histogram using the new bins values. The result
    # is an approximation of the histogram if it were collected with between values in current_bins.
    cumulative_interpolated_hist = np.interp(current_bins, bins_to_interpolate, cumulative_hist)
    # Finally, the counts of the histogram is computed based on the gap between each two adjacent bins.
    interpolated_counts = np.diff(cumulative_interpolated_hist)
    return interpolated_counts


class HistogramCollector(BaseCollector):
    """
    Collector for holding histogram of tensors going through it.
    """

    def __init__(self, n_bins: int = 2048):
        """
        Args:
            n_bins: Number of bins in the histogram.
        """

        super().__init__()
        self._n_bins = n_bins
        self._bins = None
        self._counts = None
        self._histogram_per_iteration = []

    def _merge_histograms(self):
        """
        After collecting histogram per iteration, we merge these histograms to a single histogram
        containing all samples from all iterations.
        The merge is done in a lazy manner (is computed only when actually needed).
        """
        if len(self._histogram_per_iteration) > 0:
            # Stack all bins that were gathered during inference
            bins_stack = np.vstack([hist[1] for hist in self._histogram_per_iteration])

            # The combined histogram will be computed between new min/max (which is the min/max of all histograms).
            # The bin width of the merged histogram is the minimal bin width among all histograms (to lose as less
            # information as possible during the merge).
            merged_histogram_min = np.min(bins_stack)
            merged_histogram_max = np.max(bins_stack)
            merged_histogram_bins = np.linspace(merged_histogram_min, merged_histogram_max, self._n_bins + 1)

            merged_histogram_counts = None
            for histogram in self._histogram_per_iteration:  # Iterate all collected histograms and merge them
                if merged_histogram_counts is None:  # First histogram to consider
                    merged_histogram_counts = interpolate_histogram(merged_histogram_bins, histogram[1], histogram[0])
                else:  # Merge rest of histograms into existing final histogram
                    merged_histogram_counts += interpolate_histogram(merged_histogram_bins, histogram[1], histogram[0])

            self._counts = merged_histogram_counts
            self._bins = merged_histogram_bins

    def scale(self, scale_factor: np.ndarray):
        """
        Scale all statistics in collector by some factor.
        If the scale is per-channel, the data's validity status change to invalid since histogram was collected
        per-tensor and not per-channel.

        Args:
            scale_factor: Factor to scale all collector's statistics by.

        """

        scale_per_channel = scale_factor.flatten().shape[0] > 1  # current scaling is per channel or not
        if scale_per_channel or not self.is_legal:
            self.update_legal_status(is_illegal=True)
        else:
            bins, _ = self.get_histogram()
            self._bins = bins * scale_factor

    def shift(self, shift_value: np.ndarray):
        """
        Shift all statistics in collector by some value.
        If the shifting is per-channel, the data's validity status change to invalid since histogram was collected
        per-tensor and not per-channel.

        Args:
            shift_value: Value to shift all collector's statistics by.

        """

        shift_per_channel = shift_value.flatten().shape[0] > 1  # current shifting is per channel or not
        if shift_per_channel or not self.is_legal:
            self.update_legal_status(is_illegal=True)
        else:
            bins, _ = self.get_histogram()
            self._bins = bins + shift_value

    def get_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns: The histogram (bins and counts) the collector holds.
        """

        self.validate_data_correctness()
        # If collected histograms (one per inference iteration) were not merged before, merge them and return the
        # merged histogram.
        if self._bins is None or self._counts is None:
            self._merge_histograms()
        return self._bins, self._counts

    def max(self):
        """
        Returns: Maximum value in the histogram.
        """
        bins, counts = self.get_histogram()
        return max(bins[:-1][counts > 0])

    def min(self):
        """
        Returns: Minimum value in the histogram.
        """
        bins, counts = self.get_histogram()
        return min(bins[:-1][counts > 0])

    def update(self, x: np.ndarray):
        """
        Update the current state of the histogram bins and count according to a new
        tensor that goes through the collector.

        Args:
            x: Tensor going through the collector to update the histogram according to.
        """
        count, bins = np.histogram(x, bins=self._n_bins)
        self._histogram_per_iteration.append((count, bins))