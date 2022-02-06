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

import model_compression_toolkit.common.quantization.quantization_config as qc
from model_compression_toolkit.common.constants import MIN_THRESHOLD
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_search import qparams_selection_tensor_search, qparams_selection_histogram_search
from model_compression_toolkit.common.similarity_analyzer import compute_lp_norm

from model_compression_toolkit.common.constants import THRESHOLD


def lp_selection_tensor(tensor_data: np.ndarray,
                        p: int,
                        n_bits: int,
                        per_channel: bool = False,
                        channel_axis: int = 1,
                        n_iter: int = 10,
                        min_threshold: float = MIN_THRESHOLD,
                        quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.LP) -> dict:
    """
    Compute the optimal threshold based on Lp-norm distance to quantize a tensor.
    The error is computed as the distance in Lp-norm between the tensor and its quantized
    data. The p-norm is passed as an argument, p.

    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_iter: Number of iterations to search for the optimal threshold.
        min_threshold: Minimal threshold to chose when the computed one is smaller.
        quant_error_method: an error function to optimize the parameters' selection accordingly (not used for this method).

    Returns:
        Optimal threshold to quantize the tensor based on Lp-norm distance.
    """

    def _loss(x: np.ndarray,
              y: np.ndarray,
              threshold: float) -> np.float:
        """
        Compute the Lp-norm distance between two tensors x and y.

        Args:
            x: Fake-quant quantized tensor of y.
            y: Float tensor.
            threshold: Threshold x was quantized by.

        Returns:
            The Lp-norm distance between the two tensors.
        """
        return compute_lp_norm(x, y, p)

    return {THRESHOLD: qparams_selection_tensor_search(_loss,
                                                       tensor_data,
                                                       n_bits,
                                                       per_channel=per_channel,
                                                       channel_axis=channel_axis,
                                                       n_iter=n_iter,
                                                       min_threshold=min_threshold)}


def lp_selection_histogram(bins: np.ndarray,
                           counts: np.ndarray,
                           p: int,
                           n_bits: int,
                           min_value: float,
                           max_value: float,
                           constrained=True,
                           n_iter=10,
                           min_threshold=MIN_THRESHOLD,
                           quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.LP) -> dict:
    """
    Compute the optimal threshold based on Lp-norm distance to quantize a histogram.
    The quantization error is the Euclidean distance between two points in the Lp-norm
    (the original and quantized values).

    Args:
        bins: Bins values of the histogram.
        counts: Bins counts of the histogram.
        p: p-norm to use for the Lp-norm distance.
        n_bits: Number of bits to quantize the tensor.
        min_value: Min value (not used for this method).
        max_value: Max value (not used for this method).
        constrained: Whether the threshold should be constrained or not.
        n_iter: Number of iteration ot search for the threshold.
        min_threshold: Minimal threshold to use if threshold is too small.
        quant_error_method: an error function to optimize the parameters' selection accordingly (not used for this method).

    Returns:
        Optimal threshold to quantize the histogram based on Lp-norm distance.
    """

    def _loss(q_bins: np.ndarray,
              q_count: np.ndarray,
              _bins: np.ndarray,
              _counts: np.ndarray,
              threshold: np.ndarray,  # dummy
              min_max_range: np.ndarray  # dummy
              ) -> np.float:
        """
        Compute the Lp-norm distance between two histograms.

        Args:
            q_bins: Bins values of the quantized histogram.
            q_count: Bins counts of the quantized histogram.
            _bins: Bins values of the original histogram.
            _counts: Bins counts of the original histogram.

        Returns:
            The Lp-norm distance between the two tensors histograms.
        """

        return _lp_error_histogram(q_bins,
                                   q_count,
                                   bins,
                                   counts,
                                   p)

    return {THRESHOLD: qparams_selection_histogram_search(_loss,
                                                          bins,
                                                          counts,
                                                          n_bits,
                                                          constrained=constrained,
                                                          n_iter=n_iter,
                                                          min_threshold=min_threshold)}


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

