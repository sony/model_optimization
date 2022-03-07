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


from copy import deepcopy
from typing import Tuple

import numpy as np

import model_compression_toolkit.common.quantization.quantization_config as qc
from model_compression_toolkit.common.constants import FLOAT_32, THRESHOLD
from model_compression_toolkit.common.constants import MIN_THRESHOLD
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import quantize_tensor, \
    uniform_quantize_tensor
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_search import \
    qparams_selection_tensor_search, qparams_selection_histogram_search


def kl_selection_tensor(tensor_data: np.ndarray,
                        p: int,
                        n_bits: int,
                        per_channel: bool = False,
                        channel_axis: int = 1,
                        n_iter: int = 10,
                        min_threshold: float = MIN_THRESHOLD,
                        quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.KL) -> dict:
    """
    Compute the optimal threshold based on KL-divergence to quantize the tensor.
    The error computation is based on KL-divergence to quantify the distributions
    similarity of the tensor and its quantized tensor.
    The threshold is constrained and the search is done iteratively over n_iter thresholds.

    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance (not used for this method).
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_iter: Number of iterations to search for the optimal threshold.
        min_threshold: Minimal threshold to chose when the computed one is smaller.
        quant_error_method: an error function to optimize the parameters' selection accordingly (not used for this method).

    Returns:
        Optimal threshold to quantize the tensor based on KL-divergence.
    """

    def _loss(x: np.ndarray,
              y: np.ndarray,
              threshold: float) -> np.float:
        """
        Compute the KL-divergence between two histograms of two tensors x and y.

        Args:
            x: Fake-quant quantized tensor of y.
            y: Float tensor.
            threshold: Threshold x was quantized by.

        Returns:
            The KL-divergence between the two tensors histograms.
        """
        signed = np.any(y < 0)
        range_min = 0 if not signed else -threshold
        return _kl_error_function(y, range_min=range_min, range_max=threshold, n_bits=n_bits)

    return {THRESHOLD: qparams_selection_tensor_search(_loss,
                                                       tensor_data,
                                                       n_bits,
                                                       per_channel=per_channel,
                                                       channel_axis=channel_axis,
                                                       n_iter=n_iter,
                                                       min_threshold=min_threshold)}


def kl_selection_histogram(bins: np.ndarray,
                           counts: np.ndarray,
                           p: int,
                           n_bits: int,
                           min_value: float,
                           max_value: float,
                           constrained: bool = True,
                           n_iter: int = 10,
                           min_threshold: float = MIN_THRESHOLD,
                           quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.KL) -> dict:
    """
    Compute the optimal threshold based on KL-divergence to quantize a histogram.
    The error computation is based on KL-divergence to quantify the distributions
    similarity of the histogram and its quantized histogram.

    Args:
        bins: Bins values of the histogram.
        counts: Bins counts of the histogram.
        p: p-norm to use for the Lp-norm distance (not used for this method).
        n_bits: Number of bits to quantize the tensor.
        min_value: Min value (not used for this method).
        max_value: Max value (not used for this method).
        constrained: Whether the threshold should be constrained or not.
        n_iter: Number of iteration ot search for the threshold.
        min_threshold: Minimal threshold to use if threshold is too small.
        quant_error_method: an error function to optimize the parameters' selection accordingly (not used for this method).

    Returns:
        Optimal threshold to quantize the histogram based on KL-divergence.
    """

    def _loss(q_bins: np.ndarray,
              q_count: np.ndarray,
              _bins: np.ndarray,
              _counts: np.ndarray,
              threshold,
              min_max_range  # dummy
              ) -> np.float:
        """
        Compute the KL-divergence between two histograms.

        Args:
            q_bins: Bins values of the quantized histogram.
            q_count: Bins counts of the quantized histogram.
            _bins: Bins values of the original histogram.
            _counts: Bins counts of the original histogram.
            threshold: Threshold bins was quantized by.

        Returns:
            The KL-divergence between the two tensors histograms.
        """

        return _kl_error_histogram(q_bins,
                                   q_count,
                                   _bins,
                                   _counts,
                                   range_min=-threshold,
                                   range_max=threshold)

    return {THRESHOLD: qparams_selection_histogram_search(_loss,
                                                          bins,
                                                          counts,
                                                          n_bits,
                                                          constrained=constrained,
                                                          n_iter=n_iter,
                                                          min_threshold=min_threshold)}


def _kl_error_function(x: np.ndarray,
                       range_min: float,
                       range_max: float,
                       n_bins: int = 2048,
                       n_bits: int = 8) -> np.float:
    """
    Compute the error function between a tensor to its quantized version.
    The error is computed based on the KL-divergence the distributions have.
    Number of bins to use when computing the histogram of the float tensor is passed.
    The threshold and number of bits that were used to quantize the tensor are needed to compute the
    histograms boundaries and the number of quantized bins.

    Args:
        x: Float tensor.
        range_min: min bound on the quantization range.
        range_max: max bound on the quantization range.
        n_bins: Number of bins for the float histogram.
        n_bits: Number of bits the quantized tensor was quantized by.

    Returns:
        The KL-divergence of the float histogram and the quantized histogram of the tensors.

    """
    if range_max <= range_min:
        # invalid range
        return np.inf

    # Compute the float histogram
    bc, bv = np.histogram(x, bins=n_bins)

    # If no bins are within the range return infinity.
    if not _is_range_valid(bv, range_min, range_max):
        return np.inf

    # Compute bins values of quantized histogram.
    # TODO: note that we always do uniform quantization here, since we no longer have threshold, only range
    q_bins = uniform_quantize_tensor(bv,
                                     range_min,
                                     range_max,
                                     n_bits)

    # Sum all quantized values to a single bin of that value. Other bins of the same value
    # are zero.
    bcq, _ = np.histogram(q_bins,
                          bins=bv,
                          weights=np.concatenate([bc.flatten(), np.array([0])]))

    # compute error
    return _kl_error_histogram(q_bins,
                               bcq,
                               bv,
                               bc,
                               range_min=range_min,
                               range_max=range_max)


def _kl_error_histogram(q_bins: np.ndarray,
                        q_count: np.ndarray,
                        bins: np.ndarray,
                        counts: np.ndarray,
                        range_min: float,
                        range_max: float) -> np.float:
    """
    Compute the error function between a histogram to its quantized version.
    The error is computed based on the KL-divergence the distributions have.
    If a threshold is not passed, the maximal quantized bin value is used. A minimal
    threshold can be passed.

    Args:
        q_bins: Bins values of the quantized histogram.
        q_count: Bins counts of the quantized histogram.
        bins: Bins values of the histogram.
        counts: Bins counts of the histogram.
        range_min: min bound on the quantization range.
        range_max: max bound on the quantization range.

    Returns:
        KL-divergence score between the two histograms.
    """

    if not _is_range_valid(bins, range_min, range_max):
        return np.inf

    first_bin_idx, last_bin_idx = _get_bins_indices_from_range(bins, range_min, range_max)

    if first_bin_idx == last_bin_idx:
        return 0.0

    bins_subset, counts_subset = _get_sliced_histogram(bins,
                                                       counts,
                                                       first_bin_idx,
                                                       last_bin_idx)

    q_bins_subset, q_counts_subset = _get_sliced_histogram(q_bins,
                                                           q_count,
                                                           first_bin_idx,
                                                           last_bin_idx)

    if not counts_subset.any():
        return np.inf

    counts_acc = _compute_clipped_counts(counts,
                                         counts_subset,
                                         first_bin_idx,
                                         last_bin_idx)

    qbc = np.zeros(q_counts_subset.shape)

    for qbvui in np.unique(q_bins_subset):
        q_status = q_bins_subset[:-1] == qbvui
        positive_bins = (counts_subset[q_status] > 0).astype(FLOAT_32)
        sum_relative = np.sum(counts_subset[q_status]) / (np.sum(positive_bins) + 1e-6)
        qbc[q_status] = sum_relative * positive_bins

    p_fxp = qbc / np.sum(qbc)
    p_fxp = _smooth_distribution(p_fxp)

    p_float = counts_acc / np.sum(counts_acc)
    p_float = _smooth_distribution(p_float)

    return np.sum(p_float * np.log(p_float / p_fxp))


def _get_bins_indices_from_range(bins: np.ndarray,
                                 range_min: float,
                                 range_max: float) -> Tuple[int, int]:
    """
    For bins and a threshold, compute the first and last bins in between the threshold
    ranges.

    Args:
        bins: Bins to look for its first and last bins in the range.
        range_min: min bound on the quantization range.
        range_max: max bound on the quantization range.

    Returns:
        First and last bins indices that are in a range.
    """
    assert range_min < range_max
    first_bin_idx = max(np.where(bins >= range_min)[0].min() - 1, 0)
    last_bin_idx = np.where(bins < range_max)[0].max()
    assert first_bin_idx <= last_bin_idx
    return first_bin_idx, last_bin_idx


def _is_range_valid(bins: np.ndarray, range_min: float, range_max: float) -> bool:
    """
    Check whether there are some bins from a numpy array of bins that are in between
    a threshold range or not.
    Args:
        bins: Bins to check.
        range_min: min bound on the quantization range.
        range_max: max bound on the quantization range.

    Returns:
        Whether there are bins in the range or not.
    """
    gt_range_bins = np.where(bins >= range_min)[0]
    st_range_bins = np.where(bins < range_max)[0]
    return not (len(gt_range_bins) == 0 or len(st_range_bins) == 0)


def _smooth_distribution(probability: np.ndarray) -> np.ndarray:
    """
    Smooth distribution by decreasing non-zeros counts evenly, and increasing zeros counts
    by the total amount that was decreased.
    More info: http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf

    Args:
        probability: A flatten Numpy array with probabilities.

    Returns:
        Numpy array of the smoothed distribution.
    """

    # make sure the subtracted value is smaller than all current probabilities.
    smoothing_term = np.min(probability[probability != 0]) / (2.0 * len(probability))
    assert smoothing_term > 0

    # Count the amount of zeros vs. nonzeros in the probability.
    zeros_indices = (probability == 0).astype(FLOAT_32)
    nonzeros_indices = (probability != 0).astype(FLOAT_32)
    total_indices = probability.size
    nonzero_count = nonzeros_indices.sum()
    zero_counts = total_indices - nonzero_count

    assert nonzero_count > 0, 'Can not smooth distribution that is all zeros'
    reduce_to_fix = smoothing_term * float(zero_counts) / float(nonzero_count)

    # Compute correction term
    hist = probability.astype(FLOAT_32)
    correction_hist = smoothing_term * zeros_indices + (-reduce_to_fix) * nonzeros_indices

    hist += correction_hist  # apply correction
    assert (hist <= 0).sum() == 0  # Make sure all probabilities are positive

    return hist


def _compute_clipped_counts(counts: np.ndarray,
                            counts_subset: np.ndarray,
                            first_bin_idx: int,
                            last_bin_idx: int) -> np.ndarray:
    """
    Compute a clipped Numpy array containing counts. All bins outside the range
    between first_bin_idx and last_bin_idx are clipped.
    A clipped histogram is a histogram where all bins that are out a given range
    are accumulated to the first and last bins of the bins that are in that range.

    Args:
        counts: Counts array of a histogram.
        counts_subset: Subset of the counts array of a histogram.
        first_bin_idx: First bin index of the range.
        last_bin_idx: Last bin index of the range.

    Returns:
        A clipped histogram.
    """

    p_float_accumulated_edges = deepcopy(counts_subset)
    p_float_accumulated_edges[0] += np.sum(counts[:first_bin_idx])
    p_float_accumulated_edges[-1] += np.sum(counts[last_bin_idx:])

    return p_float_accumulated_edges


def _get_sliced_histogram(bins: np.ndarray,
                          counts: np.ndarray,
                          first_bin_idx: int,
                          last_bin_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a subset of a histogram. First and last bins to consider are passed.

    Args:
        bins: Histogram's bins to create its subset bins.
        counts: Histogram's counts to create its subset counts.
        first_bin_idx: First bin index to consider.
        last_bin_idx: Last bin index to consider.

    Returns:
        A sliced histogram.
    """

    bins_subset = deepcopy(bins[first_bin_idx:last_bin_idx + 1])
    counts_subset = deepcopy(counts[first_bin_idx:last_bin_idx])

    return bins_subset, counts_subset
