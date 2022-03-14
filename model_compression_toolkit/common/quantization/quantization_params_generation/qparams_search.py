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
import itertools
from collections import Callable
from typing import Any, Tuple, Dict

import numpy as np

from model_compression_toolkit.common.constants import MIN_THRESHOLD, THRESHOLD, DEFAULT_TOL, DEFAULT_DEC_FACTOR, \
    SYMMETRIC_TENSOR_PER_CHANNEL_N_INTERVALS, SYMMETRIC_TENSOR_PER_CHANNEL_N_ITER, SYMMETRIC_TENSOR_DEC_FREQ, \
    SYMMETRIC_TENSOR_PER_CHANNEL_DEC_FREQ, SYMMETRIC_TENSOR_N_INTERVALS, SYMMETRIC_TENSOR_N_ITER, \
    UNIFORM_TENSOR_PER_CHANNEL_N_ITER, UNIFORM_TENSOR_N_ITER, SYMMETRIC_HISTOGRAM_DEC_FREQ, SYMMETRIC_HISTOGRAM_N_ITER, \
    SYMMETRIC_HISTOGRAM_N_INTERVALS, UNIFORM_HISTOGRAM_N_ITER, BOTTOM_FACTOR, UPPER_FACTOR, UNIFORM_TENSOR_N_SAMPLES, \
    UNIFORM_HISTOGRAM_N_SAMPLES, DEC_RANGE_UPPER, DEC_RANGE_BOTTOM
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import quantize_tensor, \
    reshape_tensor_for_per_channel_search, uniform_quantize_tensor, get_output_shape
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
    output_shape = get_output_shape(tensor_data.shape, channel_axis)

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
            threshold_hat = (threshold / (2 ** i)).reshape([-1, 1])
            qt = quantize_tensor(tensor_data_r, threshold_hat, n_bits, signed)
            per_channel_error = _error_function_wrapper(error_function, tensor_data_r, qt, threshold_hat)

            error_list.append(per_channel_error)
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
        q_bins = quantize_tensor(bins, threshold, n_bits, signed)  # compute the quantized values of the bins.
        error = qparams_selection_histogram_search_error_function(error_function, bins, q_bins, counts,
                                                                  threshold=threshold)
        error_list.append(error)

    # Return the threshold with the minimal error.
    return np.maximum(threshold_list[np.argmin(error_list)], min_threshold)


def qparams_symmetric_iterative_minimization(x0: np.ndarray,
                                             x: np.ndarray,
                                             loss_fn: Callable,
                                             n_bits: int,
                                             signed: bool = True,
                                             n_intervals: int = SYMMETRIC_TENSOR_N_INTERVALS,
                                             n_iter: int = SYMMETRIC_TENSOR_N_ITER,
                                             alpha: float = BOTTOM_FACTOR,
                                             beta: float = UPPER_FACTOR,
                                             dec_factor: Tuple = DEFAULT_DEC_FACTOR,
                                             dec_freq: int = SYMMETRIC_TENSOR_DEC_FREQ,
                                             tolerance: float = DEFAULT_TOL,
                                             per_channel=False) -> Dict[str, np.ndarray]:
    """
    Search for an optimal threshold to for symmetric tensor quantization.
    The search starts with the no-clipping threshold the tensor has, and continues with
    n_iter of iterative search. In each iteration, a set of n_intervals threshold is created by evenly-spacing the
    range constructed from the previous obtained threshold multiplied by the alph and beta factors (for lower and upper
    limits, respectively).
    In addition, each dec_freq iterations, the factors are multiplied by dec_factor in order to make the search range
    narrower.

        Args:
        x0: Initial threshold.
        x: Numpy array with tensor's content.
        loss_fn: Function to compute the error between the original and quantized tensors.
        n_bits: Number of bits to quantize the tensor.
        signed: Whether quantization range is signed or not.
        n_intervals: Number of locations to examine each iteration from the given range.
        n_iter: Number of searching iterations.
        alpha: Factor for creating the lower limit of the search range.
        beta: Factor for creating the upper limit of the search range.
        dec_factor: Factor for decreasing the multiplication factors, to get narrower search range.
        dec_freq: Frequency for decreasing the multiplication factors.
        tolerance: If the improvement between iterations is smaller than tolerance, then early stop.
        per_channel: Whether quantization is done per-channel or per-tensor.

    Returns:
        Dictionary with optimized threshold for symmetric tensor quantization (best obtained during the search),
        and its matching loss value.

    """
    range_scale = np.array([alpha, beta])
    curr_threshold = x0

    if per_channel:
        # wrapping loss function with per-channel wrapper for vectorized per-channel computation
        # Note: x should be already reshaped tensor for per-channel search
        curr_threshold = curr_threshold.reshape([-1, 1])
        loss = _error_function_wrapper(loss_fn, x, quantize_tensor(x, curr_threshold, n_bits, signed), curr_threshold).reshape([-1, 1])
    else:
        loss = loss_fn(x, quantize_tensor(x, curr_threshold, n_bits, signed), curr_threshold)

    best = {"param": curr_threshold, "loss": loss}

    for n in range(n_iter):
        prev_best_loss = best['loss']
        new_range_bounds = curr_threshold * range_scale

        curr_res = search_fixed_range_intervals(new_range_bounds, x, loss_fn, n_bits, signed, n_intervals, per_channel)
        curr_threshold = curr_res['param']
        curr_loss = curr_res['loss']

        imp_losses = curr_loss < prev_best_loss
        best = {"param": curr_threshold * imp_losses + best['param'] * np.logical_not(imp_losses),
                "loss": curr_loss * imp_losses + prev_best_loss * np.logical_not(imp_losses)}

        iters_loss_diff_avg = np.mean(prev_best_loss - curr_loss)
        if 0 < iters_loss_diff_avg < tolerance:
            # improvement in last step is very small on average, therefore - finishing the search
            break

        # increase scaler to make range bounds narrower in next iteration
        if n % dec_freq == 0:
            range_scale *= dec_factor
            # prevent min bound from exceeding max bound
            range_scale = np.array([min(range_scale[0], DEC_RANGE_BOTTOM), max(range_scale[1], DEC_RANGE_UPPER)])

    return best


def iterative_uniform_dynamic_range_search(x0: np.ndarray,
                                           x: np.ndarray,
                                           scalers: np.ndarray,
                                           loss_fn: Callable,
                                           n_bits: int,
                                           n_iter: int = UNIFORM_TENSOR_N_ITER,
                                           tolerance: float = DEFAULT_TOL,
                                           per_channel: bool = False) -> Dict[str, np.ndarray]:
    """
    Search for an optimal quantization range for uniform tensor quantization.
    The search starts with the no-clipping range the tensor has, and continues with
    n_iter of iterative search. In each iteration, a set of range candidates is created by multiplying each
    scaler (from the scalers set) with the current base range (the last obtained range).

        Args:
        x0: Initial base range.
        x: Numpy array with tensor's content.
        loss_fn: Function to compute the error between the original and quantized tensors.
        scalers: A set of multiplication factors, to create a set of quantization range candidates at each iteration.
        n_bits: Number of bits to quantize the tensor.
        n_iter: Number of searching iterations.
        tolerance: If the improvement between iterations is smaller than tolerance, then early stop.
        per_channel: Whether quantization is done per-channel or per-tensor.

    Returns:
        Dictionary with optimized quantization range for uniform tensor quantization (best obtained during the search),
        and its matching loss value.

    """
    curr_range_bounds = x0

    if per_channel:
        # wrapping loss function with per-channel wrapper for vectorized per-channel computation
        # Note: x should be already reshaped tensor for per-channel search and x0 is a tensor or ranges
        # of shape (num_channels, 2)
        loss = _error_function_wrapper(loss_fn, x, uniform_quantize_tensor(x,
                                                                           curr_range_bounds[:, 0].reshape([-1, 1]),
                                                                           curr_range_bounds[:, 1].reshape([-1, 1]),
                                                                           n_bits),
                                       curr_range_bounds).reshape([-1, 1])
    else:
        loss = loss_fn(x, uniform_quantize_tensor(x, curr_range_bounds[0], curr_range_bounds[1], n_bits),
                       curr_range_bounds)

    best = {"param": curr_range_bounds, "loss": loss}

    for n in range(n_iter):
        prev_best_loss = best['loss']
        curr_res = search_dynamic_range(base_range=curr_range_bounds, scalers=scalers, x=x, loss_fn=loss_fn,
                                        n_bits=n_bits, per_channel=per_channel)
        curr_range_bounds = curr_res['param']
        curr_loss = curr_res['loss']

        imp_losses = curr_loss < prev_best_loss
        best = {"param": curr_range_bounds * imp_losses + best['param'] * np.logical_not(imp_losses),
                "loss": curr_loss * imp_losses + prev_best_loss * np.logical_not(imp_losses)}

        iters_loss_diff_avg = np.mean(prev_best_loss - curr_loss)
        if 0 < iters_loss_diff_avg < tolerance:
            # improvement in last step is very small, therefore - finishing the search
            break

    return best


def search_fixed_range_intervals(range_bounds: np.ndarray,
                                 x: np.ndarray,
                                 loss_fn: Callable,
                                 n_bits: int,
                                 signed: bool = True,
                                 n_intervals: int = 100,
                                 per_channel: bool = False) -> Dict[str, np.ndarray]:
    """
    Searches in a set of n_intervals thresholds, taken from evenly-space intervales from the constructed range.

    Args:
        range_bounds: A range for creating a set of evenly-spaced threshold candidates.
        x: Numpy array with tensor's content.
        loss_fn: Function to compute the error between the original and quantized tensors.
        n_bits: Number of bits to quantize the tensor.
        signed: Whether quantization range is signed or not.
        n_intervals: Number of locations to examine each iteration from the given range.
        per_channel: Whether the search is done per-channel or per-tensor.

    Returns: Dictionary with best obtained threshold and the threshold's matching loss.

    """
    if per_channel:
        # search per-channel
        intervals = np.linspace(start=range_bounds[:, 0], stop=range_bounds[:, 1], num=n_intervals, dtype=float)
        # just the first interval values
        first_interval_thresholds = intervals[0, :].reshape([-1, 1])
        best = {"param": first_interval_thresholds,
                "loss": _error_function_wrapper(loss_fn, x,
                                                quantize_tensor(x, first_interval_thresholds, n_bits, signed),
                                                first_interval_thresholds).reshape([-1, 1])}
        for interval in range(1, n_intervals):
            interval_thresholds = intervals[interval, :].reshape([-1, 1])  # takes the i'th threshold for each channel
            interval_losses = _error_function_wrapper(loss_fn, x, quantize_tensor(x, interval_thresholds, n_bits, signed),
                                                      interval_thresholds).reshape([-1, 1])

            imp_losses = (interval_losses < best['loss']).reshape([-1, 1])

            best = {"param": interval_thresholds * imp_losses + best['param'] * np.logical_not(imp_losses),
                    "loss": interval_losses * imp_losses + best['loss'] * np.logical_not(imp_losses)}
    else:
        # search per-tensor
        intervals = np.linspace(start=range_bounds[0], stop=range_bounds[1], num=n_intervals, dtype=float)
        interval_losses = list(map(lambda t: loss_fn(x, quantize_tensor(x, t, n_bits, signed), t), intervals))
        best = {"param": intervals[np.argmin(interval_losses)], "loss": np.min(interval_losses)}

    return best


def search_dynamic_range(base_range: np.ndarray, x: np.ndarray, scalers: np.ndarray, loss_fn: Callable, n_bits: int,
                         per_channel: bool = False) -> Dict[str, np.ndarray]:
    """
    Searches in a set of constructed quantization ranges.

    Args:
        base_range: Base quantization range for constructing ranges candidates.
        x: Numpy array with tensor's content.
        scalers: A set of scale factor for constructing ranges candidates.
        loss_fn: Function to compute the error between the original and quantized tensors.
        n_bits: Number of bits to quantize the
        per_channel: Whether the search is done per-channel or per-tensor.

    Returns: Dictionary with best obtained quantization range and the threshold's matching loss.

    """
    if per_channel:
        # search per-channel
        ranges = np.stack([np.multiply.outer(base_range[:, 0], scalers[:, 0]),
                           np.multiply.outer(base_range[:, 1], scalers[:, 1])], axis=2)
        first_ranges_set = ranges[:, 0, :]
        best = {"param": first_ranges_set,
                "loss": _error_function_wrapper(loss_fn, x,
                                                uniform_quantize_tensor(x,
                                                                        first_ranges_set[:, 0].reshape([-1, 1]),
                                                                        first_ranges_set[:, 1].reshape([-1, 1]),
                                                                        n_bits),
                                                first_ranges_set).reshape([-1, 1])}
        for range_idx in range(1, ranges.shape[1]):  # iterate over sets of per-channel ranges
            ranges_set = ranges[:, range_idx, :]  # takes the i'th range for each channel
            ranges_losses = _error_function_wrapper(loss_fn, x,
                                                    uniform_quantize_tensor(x,
                                                                            ranges_set[:, 0].reshape([-1, 1]),
                                                                            ranges_set[:, 1].reshape([-1, 1]),
                                                                            n_bits),
                                                    ranges_set).reshape([-1, 1])
            imp_losses = ranges_losses < best['loss']
            best = {"param": ranges_set * imp_losses + best['param'] * np.logical_not(imp_losses),
                    "loss": ranges_losses * imp_losses + best['loss'] * np.logical_not(imp_losses)}
    else:
        # search per-tensor
        ranges = base_range * scalers
        ranges_losses = list(map(lambda mm: loss_fn(x, uniform_quantize_tensor(x, mm[0], mm[1], n_bits), mm), ranges))
        best = {"param": ranges[np.argmin(ranges_losses)], "loss": np.min(ranges_losses)}

    return best


def qparams_symmetric_selection_tensor_search(error_function: Callable,
                                              tensor_data: np.ndarray,
                                              tensor_max: np.ndarray,
                                              n_bits: int,
                                              per_channel: bool = False,
                                              channel_axis: int = 1,
                                              n_iter: int = SYMMETRIC_TENSOR_PER_CHANNEL_N_ITER,
                                              min_threshold=MIN_THRESHOLD) -> Any:
    """
    Search for optimal threshold (per-channel or per-tensor) for symmetric quantization of a tensor,
    using the iterative optimizer method.

    Args:
        error_function: Function to compute the error between the original and quantized tensors.
        tensor_data: Numpy array with tensor's content.
        tensor_max: The max value of the tensor.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the tensor should be quantized per-channel or per-tensor.
        channel_axis: Index of output channels dimension.
        n_iter: Number of searching iterations.
        min_threshold: Threshold to return if the computed threshold is smaller that min_threshold.

    Returns:
        Ndarray with an optimized threshold (or set of thresholds shaped according to the channels_axis if per-channel).

    """

    signed = np.any(tensor_data < 0)  # check if tensor is singed
    output_shape = get_output_shape(tensor_data.shape, channel_axis)

    # If the threshold is computed per-channel, we rearrange the tensor such that each sub-tensor
    # is flattened, and we iterate over each one of them when searching for the threshold.
    if per_channel:
        tensor_data_r = reshape_tensor_for_per_channel_search(tensor_data, channel_axis)
        max_tensor = np.maximum(min_threshold, tensor_max)
        res = qparams_symmetric_iterative_minimization(x0=max_tensor,
                                                       x=tensor_data_r,
                                                       loss_fn=error_function,  # gets float_tensor, fxp_tensor, threshold
                                                       n_bits=n_bits,
                                                       signed=signed,
                                                       n_intervals=SYMMETRIC_TENSOR_PER_CHANNEL_N_INTERVALS,
                                                       n_iter=SYMMETRIC_TENSOR_PER_CHANNEL_N_ITER,
                                                       dec_freq=SYMMETRIC_TENSOR_PER_CHANNEL_DEC_FREQ,
                                                       per_channel=True)
        return np.reshape(np.maximum(min_threshold, res['param']), output_shape)
    else:
        # quantize per-tensor
        res = qparams_symmetric_iterative_minimization(x0=get_init_threshold(min_threshold, tensor_max),
                                                       x=tensor_data,
                                                       loss_fn=error_function,
                                                       n_bits=n_bits,
                                                       signed=signed,
                                                       n_intervals=SYMMETRIC_TENSOR_N_INTERVALS,
                                                       n_iter=SYMMETRIC_TENSOR_N_ITER,
                                                       dec_freq=SYMMETRIC_TENSOR_DEC_FREQ,
                                                       per_channel=False)

        return max(min_threshold, res['param'])


def qparams_uniform_selection_tensor_search(error_function: Callable,
                                            tensor_data: np.ndarray,
                                            tensor_min: np.ndarray,
                                            tensor_max: np.ndarray,
                                            n_bits: int,
                                            per_channel: bool = False,
                                            channel_axis: int = 1,
                                            n_iter: int = UNIFORM_TENSOR_PER_CHANNEL_N_ITER) -> Any:
    """
    Search for optimal quantization range (per-channel or per-tensor) for uniform quantization of a tensor,
    using the iterative optimizer method and built-in scale factors
    for constructing ranges candidates during the search.

    Args:
        error_function: Function to compute the error between the original and quantized tensors.
        tensor_data: Numpy array with tensor's content.
        tensor_min: The min value of the tensor.
        tensor_max: The max value of the tensor.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the tensor should be quantized per-channel or per-tensor.
        channel_axis: Index of output channels dimension.
        n_iter: Number of searching iterations.

    Returns:
        Ndarray with an optimized range (or set of thresholds shaped according to the channels_axis if per-channel).

    """

    output_shape = get_output_shape(tensor_data.shape, channel_axis)

    alpha = np.linspace(BOTTOM_FACTOR, UPPER_FACTOR, UNIFORM_TENSOR_N_SAMPLES)
    beta = np.linspace(BOTTOM_FACTOR, UPPER_FACTOR, UNIFORM_TENSOR_N_SAMPLES)
    scalers = np.asarray(list(itertools.product(alpha, beta)))

    # If the threshold is computed per-channel, we rearrange the tensor such that each sub-tensor
    # is flattened, and we iterate over each one of them when searching for the threshold.
    if per_channel:
        if per_channel:
            tensor_data_r = reshape_tensor_for_per_channel_search(tensor_data, channel_axis)
            tensor_min_max = np.column_stack([tensor_min.flatten(), tensor_max.flatten()])
            res = iterative_uniform_dynamic_range_search(x0=tensor_min_max,
                                                         x=tensor_data_r,
                                                         scalers=scalers,
                                                         loss_fn=error_function,
                                                         n_bits=n_bits,
                                                         n_iter=UNIFORM_TENSOR_PER_CHANNEL_N_ITER,
                                                         per_channel=True)
            return np.reshape(res['param'][:, 0], output_shape), np.reshape(res['param'][:, 1], output_shape)
    else:
        # quantize per-tensor
        pass
        res = iterative_uniform_dynamic_range_search(x0=np.array([tensor_min, tensor_max]),
                                                     x=tensor_data,
                                                     scalers=scalers,
                                                     loss_fn=error_function,
                                                     n_bits=n_bits,
                                                     n_iter=UNIFORM_TENSOR_N_ITER,
                                                     per_channel=False)
        return res['param']


def qparams_symmetric_selection_histogram_search(error_function: Callable,
                                                 tensor_max: np.ndarray,
                                                 bins: np.ndarray,
                                                 counts: np.ndarray,
                                                 n_bits: int,
                                                 n_iter: int = SYMMETRIC_HISTOGRAM_N_ITER,
                                                 min_threshold: float = MIN_THRESHOLD):
    """
    search for optimal threshold (per-channel or per-tensor) for symmetric quantization of a histogram,
    using the iterative optimizer method.

    Args:
        error_function: Function to compute the error between the original and quantized histograms.
        tensor_max: The max value of the tensor.
        bins: Bins of the histogram to search_methods for an optimal threshold.
        counts: Number of elements in the bins to search_methods for a threshold.
        n_bits: Number of bits to quantize the tensor.
        n_iter: Number of searching iterations.
        min_threshold: Threshold to return if the computed threshold is smaller that min_threshold.

    Returns:
        Optimized threshold for quantifying the histogram.

    """
    signed = np.any(bins[:-1][counts != 0] < 0)  # Whether histogram contains negative values or not.

    res = qparams_symmetric_iterative_minimization(x0=get_init_threshold(min_threshold, tensor_max),
                                                   x=bins,
                                                   loss_fn=lambda x, q_x, t:  # t is dummy, we only need it for KL error
                                                   qparams_selection_histogram_search_error_function(error_function,
                                                                                                     x,
                                                                                                     q_x,
                                                                                                     counts),
                                                   n_bits=n_bits,
                                                   signed=signed,
                                                   n_intervals=SYMMETRIC_HISTOGRAM_N_INTERVALS,
                                                   n_iter=SYMMETRIC_HISTOGRAM_N_ITER,
                                                   dec_freq=SYMMETRIC_HISTOGRAM_DEC_FREQ,
                                                   per_channel=False)
    return max(min_threshold, res['param'])


def kl_qparams_symmetric_selection_histogram_search(error_function: Callable,
                                                    tensor_max: np.ndarray,
                                                    bins: np.ndarray,
                                                    counts: np.ndarray,
                                                    n_bits: int,
                                                    n_iter: int = SYMMETRIC_HISTOGRAM_N_ITER,
                                                    min_threshold: float = MIN_THRESHOLD):
    """
    Search for optimal threshold (per-channel or per-tensor) for symmetric quantization of a histogram,
    with KL-Divergence loss function (needs a separate search function
    since the error function needs additional arguments that are constructed from the input)
    Using the iterative optimizer method for the search.

    Args:
        error_function: Function to compute the error between the original and quantized histograms.
        tensor_max: The max value of the tensor.
        bins: Bins of the histogram to search_methods for an optimal threshold.
        counts: Number of elements in the bins to search_methods for a threshold.
        n_bits: Number of bits to quantize the tensor.
        n_iter: Number of searching iterations.
        min_threshold: Threshold to return if the computed threshold is smaller that min_threshold.

    Returns:
        Optimized threshold for quantifying the histogram.

    """
    signed = np.any(bins[:-1][counts != 0] < 0)  # Whether histogram contains negative values or not.
    res = qparams_symmetric_iterative_minimization(x0=get_init_threshold(min_threshold, tensor_max),
                                                   x=bins,
                                                   loss_fn=lambda x, q_x, t:
                                                   kl_qparams_selection_histogram_search_error_function(error_function,
                                                                                                        bins,
                                                                                                        q_x,
                                                                                                        counts,
                                                                                                        min_max_range=np.array(
                                                                                                            [0,
                                                                                                             t]) if not signed else np.array(
                                                                                                            [-t, t])),
                                                   n_bits=n_bits,
                                                   signed=signed,
                                                   n_intervals=SYMMETRIC_HISTOGRAM_N_INTERVALS,
                                                   n_iter=SYMMETRIC_HISTOGRAM_N_ITER,
                                                   dec_freq=SYMMETRIC_HISTOGRAM_DEC_FREQ,
                                                   per_channel=False)
    return max(min_threshold, res['param'])


def qparams_uniform_selection_histogram_search(error_function: Callable,
                                               tensor_min_max: np.ndarray,
                                               bins: np.ndarray,
                                               counts: np.ndarray,
                                               n_bits: int,
                                               n_iter: int = UNIFORM_HISTOGRAM_N_ITER):
    """
    Search for optimal quantization range (per-channel or per-tensor) for uniform quantization of a histogram,
    using the iterative optimizer method and built-in scale factors
    for constructing ranges candidates during the search.

    Args:
        error_function: Function to compute the error between the original and quantized histograms.
        tensor_min_max: Numpy array with tensor's min and max values.
        bins: Bins of the histogram to search_methods for an optimal threshold.
        counts: Number of elements in the bins to search_methods for a threshold.
        n_bits: Number of bits to quantize the tensor.
        n_iter: Number of searching iterations.

    Returns:
        Optimized range for quantifying the histogram.

    """
    alpha = np.linspace(BOTTOM_FACTOR, UPPER_FACTOR, UNIFORM_HISTOGRAM_N_SAMPLES)
    beta = np.linspace(BOTTOM_FACTOR, UPPER_FACTOR, UNIFORM_HISTOGRAM_N_SAMPLES)
    scalers = np.asarray(list(itertools.product(alpha, beta)))
    res = iterative_uniform_dynamic_range_search(x0=tensor_min_max,
                                                 x=bins,
                                                 scalers=scalers,
                                                 loss_fn=lambda x, q_x, mm:
                                                 qparams_selection_histogram_search_error_function(error_function,
                                                                                                   x,
                                                                                                   q_x,
                                                                                                   counts,
                                                                                                   min_max_range=mm),
                                                 n_bits=n_bits,
                                                 n_iter=UNIFORM_HISTOGRAM_N_ITER,
                                                 per_channel=False)
    return res['param']


def qparams_selection_histogram_search_error_function(error_function: Callable,
                                                      bins: np.ndarray,
                                                      q_bins: np.ndarray,
                                                      counts: np.ndarray,
                                                      threshold: np.ndarray = None,
                                                      min_max_range=None):
    """
    Computes the error according to the given error function, to be used in the parameters' selection process
    for quantization.
    Args:
        error_function: Function to compute the error between the original and quantized histograms.
        bins: Bins values of the histogram.
        q_bins: Bins values of the quantized histogram.
        counts: Bins counts of the original histogram.
        threshold: Threshold bins were quantized by (used only for kl error function).
        min_max_range: quantization parameter, used in uniform parameters' selection for quantization range validation
        (not used for symmetric parameters' selection)

    Returns: the error between the original and quantized histogram.

    """
    # computes the number of elements between quantized bin values.
    q_count, _ = np.histogram(q_bins, bins=bins, weights=np.concatenate([counts.flatten(), np.asarray([0])]))
    # threshold is only used for KL error method calculations.
    # other error methods are passed with a wrapper that accepts a threshold argument but does not use it.
    error = error_function(q_bins, q_count, bins, counts, threshold, min_max_range)  # computes the error
    return error


def kl_qparams_selection_histogram_search_error_function(error_function: Callable,
                                                         bins: np.ndarray,
                                                         q_bins: np.ndarray,
                                                         counts: np.ndarray,
                                                         min_max_range: np.ndarray):
    """
    Computes the error according to the KL-divergence the distributions of the given histogram.
    The error value is used in the threshold optimization process, for symmetric quantization.
    Args:
        error_function: Function to compute the error between the original and quantized histograms.
        bins: Bins values of the histogram.
        q_bins: Quantized bins values of the histogram.
        counts: Bins counts of the original histogram.
        min_max_range: Quantization range to quantize histogram by.

    Returns: the error between the original and quantized histogram.

    """
    # compute the number of elements between quantized bin values.
    q_count, _ = np.histogram(q_bins, bins=bins, weights=np.concatenate([counts.flatten(), np.asarray([0])]))
    error = error_function(q_bins, q_count, bins, counts, range_min=min_max_range[0], range_max=min_max_range[1])
    return error


def get_init_threshold(min_threshold: float, tensor_max: np.ndarray, per_channel: bool = False) -> np.ndarray:
    """
    Gets an initial value for the threshold optimization process.
    If per_channel then returns a vector with initial value for each channel.

    Args:
        min_threshold: Minimal threshold to use if threshold is too small (not used for this method).
        tensor_max: Max value of a tensor.
        per_channel: Whether the quantization should be per-channel or not.

    Returns:
        Threshold value if max value in tensor is larger than min_threshold.
    """
    if per_channel:
        init_t = tensor_max
        init_t[tensor_max < min_threshold] = min_threshold
        return init_t
    return max(min_threshold, tensor_max)


def _error_function_wrapper(error_function: Callable,
                            float_tensor: np.ndarray,
                            q_tensor: np.ndarray,
                            in_params: np.ndarray) -> np.ndarray:
    """
    Wrapper method for using the error methods in a vectorized per-channel parameters' search.

    Args:
        error_function: error_function: Function to compute the error between the original and quantized tensors.
        float_tensor: Numpy array with float tensor's content.
        q_tensor: Numpy array with quantized tensor's content.
        in_params: Quantization params the tensor is quantized by (used in specific error functions only).

    Returns: A list of error values per-channel for the quantized tensor, according to the error function.
    """
    _error_per_list = []
    for j in range(float_tensor.shape[0]):  # iterate all channels of the tensor.
        _error_per_list.append(error_function(float_tensor[j, :], q_tensor[j, :], in_params[j]))
    return np.asarray(_error_per_list)
