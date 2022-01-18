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

from collections import Callable
from typing import Any
from scipy import optimize

import numpy as np

from model_compression_toolkit.common.constants import MIN_THRESHOLD, THRESHOLD
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import quantize_tensor, \
    reshape_tensor_for_per_channel_search, uniform_quantize_tensor, fix_range_to_include_zero
from model_compression_toolkit.common.quantization.quantization_params_generation.no_clipping import \
    no_clipping_selection_tensor, no_clipping_selection_histogram


def qparams_selection_tensor_search(error_function: Callable,
                                    tensor_data: np.ndarray,
                                    n_bits: int,
                                    per_channel: bool = False,
                                    channel_axis: int = 1,
                                    n_iter: int = 10,
                                    min_threshold=MIN_THRESHOLD) -> Any:
    """
    Search for an optimal threshold to quantize a tensor.
    The search_methods starts with the constrained no-clipping threshold the tensor has, and continues with
    n_iter another smaller constrained thresholds. For each candidate threshold, an error is computed
    based on the passed error function, and the threshold which yields the minimal error is selected
    and returned.

    Args:
        error_function: Function to compute the error between the original and quantized tensors.
        tensor_data: Numpy array with tensor's content.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the tensor should be quantized per-channel or per-tensor.
        channel_axis: Index of output channels dimension.
        n_iter: Number of searching iterations.
        min_threshold: Threshold to return if the computed threshold is smaller that min_threshold.

    Returns:
        Optimal constrained threshold to quantize the tensor.

    """

    signed = np.any(tensor_data < 0)  # check if tensor is singed
    output_shape = [-1 if i is channel_axis else 1 for i in range(len(tensor_data.shape))]

    # First threshold to check is the constrained threshold based on the tensor's maximal value.
    threshold = 2 * no_clipping_selection_tensor(tensor_data,
                                                 0,
                                                 n_bits,
                                                 per_channel,
                                                 channel_axis,
                                                 min_threshold=min_threshold)[THRESHOLD]

    # If the threshold is computed per-channel, we rearrange the tensor such that each sub-tensor
    # is flattened, and we iterate over each one of them when searching for the threshold.
    if per_channel:
        tensor_data_r = reshape_tensor_for_per_channel_search(tensor_data, channel_axis)

    error_list = []  # init an empty error list
    # On each iteration a new constrained threshold which equal to half of the previous tested threshold
    # is used for quantizing the tensor and computing the error. The error is appended to an error list, which
    # eventually used to select the threshold with the minimal error.
    for i in range(n_iter):
        if per_channel:
            per_channel_error = []
            for j in range(tensor_data_r.shape[0]):  # iterate all channels of the tensor.
                qt = quantize_tensor(tensor_data_r[j, :], threshold.flatten()[j] / (2 ** i), n_bits, signed)
                error = error_function(qt, tensor_data_r[j, :], threshold=threshold.flatten()[j] / (2 ** i))
                per_channel_error.append(error)
            error_list.append(np.asarray(per_channel_error))
        else:  # quantize per-tensor
            qt = quantize_tensor(tensor_data, threshold / (2 ** i), n_bits, signed)
            error = error_function(qt, tensor_data, threshold=threshold / (2 ** i))
            error_list.append(error)

    # Take the index of the minimal error, and use it compute the threshold which yielded it.
    i = np.argmin(np.stack(error_list, axis=-1), axis=-1)

    return np.maximum(np.reshape(threshold.flatten() / np.power(2, i), output_shape), min_threshold)


def qparams_selection_histogram_search(error_function: Callable,
                                       bins: np.ndarray,
                                       counts: np.ndarray,
                                       n_bits: int,
                                       constrained: bool = True,
                                       n_iter: int = 10,
                                       min_threshold: float = MIN_THRESHOLD):
    """
    Search for an optimal threshold to quantize a histogram of collected float values.
    The search_methods starts with the constrained no-clipping threshold by the bins' maximal value, and continues with
    n_iter another smaller constrained thresholds. For each candidate threshold, an error is computed
    based on the passed error function, and the threshold which yields the minimal error is selected
    and returned.

    Args:
        error_function: Function to compute the error between the original and quantized histograms.
        bins: Bins of the histogram to search_methods for an optimal threshold.
        counts: Number of elements in the bins to search_methods for a threshold.
        n_bits: Number of bits to quantize the tensor.
        constrained: Whether the threshold should be constrained or not.
        n_iter: Number of searching iterations.
        min_threshold: Threshold to return if the computed threshold is smaller that min_threshold.

    Returns:
        Optimal constrained threshold to quantize the tensor.

    """

    signed = np.any(bins < 0)  # Whether histogram contains negative values or not.
    threshold = (1 + int(constrained)) * no_clipping_selection_histogram(bins,
                                                                         counts,
                                                                         p=0,#dummy
                                                                         n_bits=n_bits,#dummy
                                                                         min_value=0,#dummy
                                                                         max_value=0,#dummy
                                                                         constrained=constrained,
                                                                         n_iter=n_iter, #dummy
                                                                         min_threshold=min_threshold)
    # Init a list of thresholds.
    error_list = []
    threshold_list = threshold / np.power(2, np.linspace(0, n_iter - 1, n_iter))

    # On each iteration a new constrained threshold which equal to half of the previous tested threshold
    # is used for quantizing the histogram and computing the error. The error is appended to an error list, which
    # eventually used to select the threshold with the minimal error.
    for threshold in threshold_list:
        error = qparams_selection_histogram_search_error_function(error_function, bins, threshold, n_bits,
                                                                  signed, counts)
        error_list.append(error)

    # Return the threshold with the minimal error.
    return np.maximum(threshold_list[np.argmin(error_list)], min_threshold)


def qparams_selection_histogram_search_error_function(error_function: Callable,
                                                      bins: np.ndarray,
                                                      threshold: np.ndarray,
                                                      n_bits: int,
                                                      signed: bool,
                                                      counts: np.ndarray):
    """
    Calculates the error according to the given error function, to be used in the threshold optimization process
    for symmetric and power-of-two quantization.
    Args:
        error_function: Function to compute the error between the original and quantized histograms.
        bins: Bins values of the histogram.
        threshold: Threshold for quantization ranges.
        n_bits: Number of bits to quantize the tensor.
        signed: Whether the tensor contains negative values or not.
        counts: Bins counts of the original histogram.

    Returns: the error between the original and quantized histogram.

    """
    q_bins = quantize_tensor(bins, threshold, n_bits, signed)  # compute the quantized values of the bins.
    # compute the number of elements between quantized bin values.
    q_count, _ = np.histogram(q_bins, bins=bins, weights=np.concatenate([counts.flatten(), np.asarray([0])]))
    error = error_function(q_bins, q_count, bins, counts)  # compute the error
    return error


def uniform_qparams_selection_histogram_search_error_function(error_function: Callable,
                                                              bins: np.ndarray,
                                                              min_max_range: np.ndarray,
                                                              n_bits: int,
                                                              counts: np.ndarray):
    """
    Calculates the error according to the given error function, to be used in the threshold optimization process
    for symmetric quantization.
    Args:
        error_function:
        bins: Bins values of the histogram.
        min_max_range: an array containing minimum and maximum values (arrays as well) for the quantization range.
        n_bits: Number of bits to quantize the tensor.
        counts: Bins counts of the original histogram.

    Returns: the error between the original and quantized histogram.

    """
    # TODO: argument 'q_bins' is only relevant for 'kl_selection' but still needs to be provided to all others.
    #       consider remove its calculation if unnecessary (and then this method is not needed, can just
    #       pass 'error_function' on 'quantize_tensor' result to the lambda in the selection method above).

    min_max_range = fix_range_to_include_zero(min_max_range, n_bits)

    q_bins = uniform_quantize_tensor(bins, range_min=min_max_range[0], range_max=min_max_range[1],
                                     n_bits=n_bits)  # compute the quantized values of the bins.
    # compute the number of elements between quantized bin values.
    q_count, _ = np.histogram(q_bins, bins=bins, weights=np.concatenate([counts.flatten(), np.asarray([0])]))
    error = error_function(q_bins, q_count, bins, counts)  # compute the error
    return error


def symmetric_qparams_tensor_minimization(x, x0, n_bits, signed, error_function):
    """
    Search for an optimal threshold to quantize a tensor.
    Uses scipy.optimization.minimize method to search through the space of possible
    unconstrained thresholds for symmetric quantization.

    Args:
        x: Numpy array with tensor's content.
        x0: An initial solution guess for the minimization process.
        n_bits: Number of bits to quantize the tensor.
        signed: Whether the tensor contains negative values or not.
        error_function: Function to compute the error between the original and quantized tensors.

    Returns:
        Optimal threshold to quantize the tensor.

    """
    return optimize.minimize(fun=lambda threshold: error_function(x, quantize_tensor(x, threshold, n_bits, signed)),
                             x0=x0)


def symmetric_qparams_histogram_minimization(x, x0, n_bits, signed, counts, error_function):
    """
        Search for an optimal threshold to quantize a histogram.
        Uses scipy.optimization.minimize method to search through the space of possible
        unconstrained thresholds for symmetric quantization.

        Args:
            x: Numpy array with tensor's content.
            x0: An initial solution guess for the minimization process.
            n_bits: Number of bits to quantize the tensor.
            signed: Whether the tensor contains negative values or not.
            counts: Number of elements in the bins to search_methods for a threshold.
            error_function: Function to compute the error between the original and quantized tensors.

        Returns:
            Optimal threshold to quantize the histogram.

    """
    return optimize.minimize(
        fun=lambda threshold: qparams_selection_histogram_search_error_function(error_function, x, threshold,
                                                                                n_bits, signed, counts),
        x0=x0)


def uniform_qparams_tensor_minimization(x, x0, n_bits, error_function):
    """
        Search for an optimal quantization range to quantize a tensor.
        Uses scipy.optimization.minimize method to search through the space of possible ranges for uniform quantization.

        Args:
            x: Numpy array with tensor's content.
            x0: An initial solution guess for the minimization process.
            n_bits: Number of bits to quantize the tensor.
            error_function: Function to compute the error between the original and quantized tensors.

        Returns:
            Optimal quantization range to quantize the tensor.

    """
    return optimize.minimize(
        fun=lambda min_max_range:
        error_function(x, uniform_quantize_tensor(x, min_max_range[0], min_max_range[1], n_bits,)),
        x0=x0)


def uniform_qparams_histogram_minimization(x, x0, n_bits, counts, error_function):
    """
        Search for an optimal quantization range to quantize a histogram.
        Uses scipy.optimization.minimize method to search through the space of possible ranges for uniform quantization.

        Args:
            x: Numpy array with tensor's content.
            x0: An initial solution guess for the minimization process.
            n_bits: Number of bits to quantize the tensor.
            counts: Number of elements in the bins to search_methods for a threshold.
            error_function: Function to compute the error between the original and quantized tensors.

        Returns:
            Optimal quantization range to quantize the histogram.

    """
    return optimize.minimize(
        fun=lambda min_max_range: uniform_qparams_selection_histogram_search_error_function(error_function, x,
                                                                                            min_max_range, n_bits,
                                                                                            counts),
        x0=x0)
