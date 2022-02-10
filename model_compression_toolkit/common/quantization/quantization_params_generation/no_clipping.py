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
from model_compression_toolkit.common.constants import MIN_THRESHOLD, THRESHOLD
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import power_of_two_constraint, \
    get_tensor_max


def no_clipping_selection_tensor(tensor_data: np.ndarray,
                                 p: int,
                                 n_bits: int,
                                 per_channel: bool = False,
                                 channel_axis: int = 1,
                                 min_threshold: float = MIN_THRESHOLD,
                                 quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.NOCLIPPING) -> dict:
    """
    Compute the constrained threshold of a tensor using the tensor's maximal value.
    If per_channel is True, multiple constrained thresholds will return.

    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance (not used for this method).
        n_bits: Number of bits to quantize the tensor (not used for this method).
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_iter: Number of iterations to search for the optimal threshold.
        min_threshold: Minimal threshold to chose when the computed one is smaller.
        quant_error_method: an error function to optimize the parameters' selection accordingly (not used for this method).

    Returns:
        Constrained no-clipping threshold to quantize the tensor.

    """

    tensor_data = np.abs(tensor_data)
    tensor_max = get_tensor_max(tensor_data, per_channel, channel_axis)

    return {THRESHOLD: power_of_two_constraint(tensor_max, min_threshold)}


def no_clipping_selection_histogram(bins: np.ndarray,
                                    counts: np.ndarray,
                                    p: int,
                                    n_bits: int,
                                    min_value: float,
                                    max_value: float,
                                    constrained: bool = True,
                                    n_iter: int = 10,
                                    min_threshold: float = MIN_THRESHOLD,
                                    quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.NOCLIPPING) -> np.ndarray:
    """
    Compute a threshold based on a histogram. The threshold can be either constrained or unconstrained.
    If computed threshold is less than min_threshold, min_threshold is returned.

    Args:
        bins: Bins values of the histogram.
        counts: Bins counts of the histogram (not used for this method).
        p: p-norm to use for the Lp-norm distance (not used for this method).
        n_bits: Number of bits to quantize the tensor (not used for this method).
        min_value: Min value (not used for this method).
        max_value: Max value (not used for this method).
        constrained: Whether the threshold should be constrained or not.
        n_iter: Number of iteration ot search for the threshold (not used for this method).
        min_threshold: Minimal threshold to use if threshold is too small.
        quant_error_method: an error function to optimize the parameters' selection accordingly (not used for this method).

    Returns:
        Threshold of a histogram.
    """

    tensor_data = np.abs(bins)
    tensor_max = np.max(tensor_data)
    if not constrained:
        return tensor_max
    return power_of_two_constraint(tensor_max, min_threshold=min_threshold)


def no_clipping_selection_min_max(bins: np.ndarray,
                                  counts: np.ndarray,
                                  p: int,
                                  n_bits: int,
                                  min_value: float,
                                  max_value: float,
                                  constrained: bool = True,
                                  n_iter: int = 10,
                                  min_threshold: float = MIN_THRESHOLD,
                                  quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.NOCLIPPING) -> dict:
    """
    Get a constrained threshold between min and max numbers.
    If computed threshold is less than min_threshold, min_threshold is returned.

    Args:
        bins: Bins values of the histogram (not used for this method).
        counts: Bins counts of the histogram (not used for this method).
        p: p-norm to use for the Lp-norm distance (not used for this method).
        n_bits: Number of bits to quantize the tensor (not used for this method).
        min_value: Min value.
        max_value: Max value.
        constrained: Whether the threshold should be constrained or not (not used for this method).
        n_iter: Number of iteration ot search for the threshold (not used for this method).
        min_threshold: Minimal threshold to use if threshold is too small.
        quant_error_method: an error function to optimize the parameters' selection accordingly (not used for this method).

    Returns:
        A constrained threshold of the min/max values.
    """
    return {THRESHOLD: no_clipping_selection_histogram(np.asarray([min_value, max_value]),
                                                                  counts,
                                                                  p,
                                                                  n_bits,
                                                                  min_value,
                                                                  max_value,
                                                                  constrained,
                                                                  n_iter,
                                                                  min_threshold=min_threshold)}
