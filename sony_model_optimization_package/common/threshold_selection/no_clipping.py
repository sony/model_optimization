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

from sony_model_optimization_package.common.constants import MIN_THRESHOLD, THRESHOLD
from sony_model_optimization_package.common.quantization.quantizers.quantizers_helpers import power_of_two_constraint


def no_clipping_selection_tensor(tensor_data: np.ndarray,
                                 p: int,
                                 n_bits: int,
                                 per_channel: bool = False,
                                 channel_axis: int = 1,
                                 min_threshold: float = MIN_THRESHOLD) -> dict:
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

    Returns:
        Constrained no-clipping threshold to quantize the tensor.

    """

    tensor_data = np.abs(tensor_data)
    if per_channel:
        output_shape = [-1 if i is channel_axis else 1 for i in range(len(tensor_data.shape))]
        # rearrange the shape indices for transposing the tensor
        shape_index = [channel_axis, *[i for i in range(len(tensor_data.shape)) if i is not channel_axis]]
        # New shape of the tensor after transposing it and reshape it
        new_shape = [tensor_data.shape[channel_axis], -1]
        tensor_data_t = np.transpose(tensor_data, shape_index)
        tensor_data = np.reshape(tensor_data_t, new_shape)
        tensor_max = np.reshape(np.max(tensor_data, axis=-1), output_shape)
    else:
        tensor_max = np.max(tensor_data)

    return {THRESHOLD: power_of_two_constraint(tensor_max, min_threshold)}


def no_clipping_selection_histogram(bins: np.ndarray,
                                    counts: np.ndarray,
                                    p: int,
                                    n_bits: int,
                                    min_value: float,
                                    max_value: float,
                                    constrained: bool = True,
                                    n_iter: int = 10,
                                    min_threshold: float = MIN_THRESHOLD) -> np.ndarray:
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
                                  min_threshold: float = MIN_THRESHOLD) -> dict:
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
