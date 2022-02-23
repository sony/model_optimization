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
    reshape_tensor_for_per_channel_search, uniform_quantize_tensor, get_output_shape, get_range_bounds, \
    get_threshold_bounds
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
        q_bins = quantize_tensor(bins, threshold, n_bits, signed)  # compute the quantized values of the bins.
        error = qparams_selection_histogram_search_error_function(error_function, bins, q_bins, counts, threshold=threshold)
        error_list.append(error)

    # Return the threshold with the minimal error.
    return np.maximum(threshold_list[np.argmin(error_list)], min_threshold)


def qparams_tensor_minimization(x, x0, error_function, quant_function, bounds=None):
    """
        Search for an optimal quantization parameters to quantize a tensor.
        Uses scipy.optimization.minimize method to search through the space of possible parameters
        to quantize the tensor according to the given quantization function and possible parameters bounds.
        Used for parameters' selection search for Symmetric and Uniform quantization
        (depends on quant_function argument).

        Args:
            x: Numpy array with tensor's content.
            x0: An initial solution guess for the minimization process.
            error_function: Function to compute the error between the original and quantized tensors.
            quant_function: Function to quantize the tensor.
            bounds: Lower and upper bounds for the quantization parameters' values
            (threshold for symmetric quantization and range for uniform quantization).

        Returns:
            Optimal quantization range to quantize the tensor.

        """
    return optimize.minimize(fun=lambda qparam: error_function(x, quant_function(qparam), qparam),
                             x0=x0,
                             bounds=bounds,
                             method='Nelder-Mead')


def qparams_histogram_minimization(x, x0, counts, error_function, quant_function, bounds=None):
    """
        Search for an optimal parameters to quantize a histogram.
        Uses scipy.optimization.minimize method to search through the space of possible
        parameters for quantization according to the given quantization method and possible parameters bounds.
        Used for parameters' selection search for Symmetric and Uniform quantization
        (depends on quant_function argument).

        Args:
            x: Numpy array with tensor's content.
            x0: An initial solution guess for the minimization process.
            counts: Number of elements in the bins to search_methods for a threshold.
            error_function: Function to compute the error between the original and quantized tensors.
            quant_function: Function to quantize the tensor.
            bounds: Lower and upper bounds for the quantization parameters' values
            (threshold for symmetric quantization and range for uniform quantization).


        Returns:
            OptimizeResult object containing the optimal parameters to quantize the histogram by.

    """
    return optimize.minimize(fun=lambda qparam:
                             qparams_selection_histogram_search_error_function(error_function=error_function,
                                                                               bins=x,
                                                                               q_bins=quant_function(qparam),
                                                                               counts=counts,
                                                                               min_max_range=qparam),
                             x0=x0,
                             bounds=bounds,
                             method='Nelder-Mead')


def kl_symmetric_qparams_histogram_minimization(x, x0, counts, n_bits, signed, error_function, bounds):
    """
    Search for an optimal threshold to quantize a histogram, using the KL error method.
    Uses scipy.optimization.minimize method to search through the space of possible
    parameters for symmetric quantization.

    Args:
        x: Numpy array with tensor's content.
        x0: An initial solution guess for the minimization process.
        counts: Number of elements in the bins to search_methods for a threshold.
        n_bits: Number of bits to quantize the tensor.
        signed: Whether the quantization range should include negative values or not.
        error_function: Function to compute the error between the original and quantized tensors.
        bounds: Lower and upper bounds for the quantization parameters' values
            (threshold for symmetric quantization and range for uniform quantization).


    Returns:
        OptimizeResult object containing the optimal threshold to quantize the histogram by.

    """
    return optimize.minimize(fun=lambda threshold:
                             kl_qparams_selection_histogram_search_error_function(error_function=error_function,
                                                                                  bins=x,
                                                                                  q_bins=quantize_tensor(x,
                                                                                                         threshold,
                                                                                                         n_bits,
                                                                                                         signed),
                                                                                  counts=counts,
                                                                                  min_max_range=np.array([0, threshold]) if not signed else np.array([-threshold, threshold])),
                             x0=x0,
                             bounds=bounds,
                             method='Nelder-Mead')


def kl_uniform_qparams_histogram_minimization(x, x0, counts, n_bits, error_function, bounds=None):
    """
    Search for an optimal range to quantize a histogram, using the KL error method.
    Uses scipy.optimization.minimize method to search through the space of possible
    parameters for uniform quantization.

    Args:
        x: Numpy array with tensor's content.
        x0: An initial solution guess for the minimization process.
        counts: Number of elements in the bins to search_methods for a threshold.
        n_bits: Number of bits to quantize the tensor.
        error_function: Function to compute the error between the original and quantized tensors.
        bounds: Lower and upper bounds for the quantization parameters' values
            (threshold for symmetric quantization and range for uniform quantization).


    Returns:
        OptimizeResult object containing the optimal threshold to quantize the histogram by.

    """
    return optimize.minimize(fun=lambda min_max_range:
                             kl_qparams_selection_histogram_search_error_function(error_function=error_function,
                                                                                  bins=x,
                                                                                  q_bins=uniform_quantize_tensor(x,
                                                                                                                 range_min=min_max_range[0],
                                                                                                                 range_max=min_max_range[1],
                                                                                                                 n_bits=n_bits),
                                                                                  counts=counts,
                                                                                  min_max_range=min_max_range),
                             x0=x0,
                             bounds=bounds,
                             method='Nelder-Mead')


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


def symmetric_qparams_selection_per_channel_search(tensor_data, tensor_max, channel_axis, search_function, min_threshold):
    """
    A wrapper function for running a search for optimal threshold per-channel of a tensor,
    to be used in symmetric quantization.

    Args:
        tensor_data: Numpy array with tensor's content.
        tensor_max: The maximal values per channel.
        channel_axis: Index of output channels dimension.
        search_function: Function to preform the parameters' optimization for a given channel.
        min_threshold: Threshold to return if the computed threshold is smaller that min_threshold.

    Returns: ndarray with the optimal threshold for each channel, shaped according to the channels_axis.

    """
    tensor_data_r = reshape_tensor_for_per_channel_search(tensor_data, channel_axis)
    output_shape = get_output_shape(tensor_data.shape, channel_axis)
    res = []

    for j in range(tensor_data_r.shape[0]):  # iterate all channels of the tensor.
        channel_data = tensor_data_r[j, :]
        channel_threshold = max(min_threshold, tensor_max.flatten()[j])

        bounds = get_threshold_bounds(min_threshold, channel_threshold)
        channel_res = search_function(channel_data, channel_threshold, bounds)
        res.append(channel_res.x)
    return np.reshape(np.array(res), output_shape)


def uniform_qparams_selection_per_channel_search(tensor_data, tensor_min, tensor_max, channel_axis, search_function):
    """
    A wrapper function for running a search for optimal range per channel of a tensor,
    to be used in uniform quantization.

    Args:
        tensor_data: Numpy array with tensor's content.
        tensor_min: The minimal values per channel.
        tensor_max: The maximal values per channel.
        channel_axis: Index of output channels dimension.
        search_function: Function to preform the parameters' optimization for a given channel.

    Returns: ndarray with the optimal threshold for each channel, shaped according to the channels_axis.

    """
    tensor_data_r = reshape_tensor_for_per_channel_search(tensor_data, channel_axis)
    output_shape = get_output_shape(tensor_data.shape, channel_axis)
    res_min, res_max = [], []

    for j in range(tensor_data_r.shape[0]):  # iterate all channels of the tensor.
        channel_data = tensor_data_r[j, :]
        channel_range_min = tensor_min.flatten()[j]
        channel_range_max = tensor_max.flatten()[j]
        channel_min_max = np.array([channel_range_min, channel_range_max])

        bounds = get_range_bounds(channel_range_min, channel_range_max)
        channel_res = search_function(channel_data, channel_min_max, bounds)

        res_min.append(channel_res.x[0])
        res_max.append(channel_res.x[1])

    res_min = np.reshape(np.array(res_min), output_shape)
    res_max = np.reshape(np.array(res_max), output_shape)
    return res_min, res_max
