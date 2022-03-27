# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from typing import Callable

import numpy as np

import model_compression_toolkit.common.quantization.quantization_config as qc
from model_compression_toolkit.common.constants import MIN_THRESHOLD, RANGE_MIN, RANGE_MAX
from model_compression_toolkit.common.quantization.quantization_params_generation.kl_selection import \
    _kl_error_histogram, _kl_error_function
from model_compression_toolkit.common.quantization.quantization_params_generation.lp_selection import \
    _lp_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mae_selection import \
    _mae_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mse_selection import \
    _mse_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_search import \
    qparams_uniform_selection_tensor_search, qparams_uniform_selection_histogram_search
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import get_tensor_max, \
    get_tensor_min

from model_compression_toolkit.common.similarity_analyzer import compute_mse, compute_mae, compute_lp_norm


def uniform_selection_tensor(tensor_data: np.ndarray,
                             p: int,
                             n_bits: int,
                             per_channel: bool = False,
                             channel_axis: int = 1,
                             n_iter: int = 10,
                             min_threshold: float = MIN_THRESHOLD,
                             quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE) -> dict:
    """
    Compute the optimal quantization range based on the provided QuantizationErrorMethod
    to uniformly quantize the tensor.
    Different search is applied, depends on the value of the selected QuantizationErrorMethod.

    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_iter: Number of iterations to search for the optimal threshold (not used for this method).
        min_threshold: Minimal threshold to use if threshold is too small (not used for this method).
        quant_error_method: an error function to optimize the range parameters' selection accordingly.

    Returns:
        Optimal quantization range to quantize the tensor uniformly.
    """
    tensor_min = get_tensor_min(tensor_data, per_channel, channel_axis)
    tensor_max = get_tensor_max(tensor_data, per_channel, channel_axis)

    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        mm = tensor_min, tensor_max
    else:
        error_function = get_range_selection_tensor_error_function(quant_error_method, p, norm=False)
        mm = qparams_uniform_selection_tensor_search(error_function,
                                                     tensor_data,
                                                     tensor_min,
                                                     tensor_max,
                                                     n_bits,
                                                     per_channel,
                                                     channel_axis)
    return {RANGE_MIN: mm[0],
            RANGE_MAX: mm[1]}


def uniform_selection_histogram(bins: np.ndarray,
                                counts: np.ndarray,
                                p: int,
                                n_bits: int,
                                min_value: float,
                                max_value: float,
                                constrained: bool = True,
                                n_iter: int = 20,
                                min_threshold: float = MIN_THRESHOLD,
                                quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE) -> dict:
    """
    Compute the optimal quantization range based on the provided QuantizationErrorMethod
    to uniformly quantize the histogram.
    Different search is applied, depends on the value of the selected QuantizationErrorMethod.

    Args:
        bins: Bins values of the histogram.
        counts: Bins counts of the histogram.
        p: p-norm to use for the Lp-norm distance (used only for lp threshold selection).
        n_bits: Number of bits to quantize the tensor.
        min_value: Min value (not used for this method).
        max_value: Max value (not used for this method).
        constrained: Whether the threshold should be constrained or not (not used for this method).
        n_iter: Number of iteration ot search for the threshold (not used for this method).
        min_threshold: Minimal threshold to use if threshold is too small (not used for this method).
        quant_error_method: an error function to optimize the range parameters selection accordingly.

    Returns:
        Optimal quantization range to quantize the histogram uniformly.
    """
    tensor_min = np.min(bins[:-1][counts > 0])
    tensor_max = np.max(bins[1:][counts > 0])
    tensor_min_max = np.array([tensor_min, tensor_max])

    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        mm = tensor_min_max
    else:
        error_function = get_range_selection_histogram_error_function(quant_error_method, p)
        mm = qparams_uniform_selection_histogram_search(error_function,
                                                        tensor_min_max,
                                                        bins,
                                                        counts,
                                                        n_bits)

    return {RANGE_MIN: mm[0],
            RANGE_MAX: mm[1]}


def uniform_no_clipping_selection_min_max(bins: np.ndarray,
                                          counts: np.ndarray,
                                          p: int,
                                          n_bits: int,
                                          min_value: float,
                                          max_value: float,
                                          constrained: bool = False,
                                          n_iter: int = 20,
                                          min_threshold: float = MIN_THRESHOLD,
                                          quant_error_method: qc.QuantizationErrorMethod =
                                          qc.QuantizationErrorMethod.NOCLIPPING) -> dict:
    """
    Gets a quantization rage between min and max numbers.

    Returns:
        A constrained threshold of the min/max values.
    """
    return uniform_selection_histogram(np.asarray([min_value, max_value]),  # histogram with min-max values only
                                       np.asarray([1]),  # passing dummy counts just to make the dimensions work
                                       p,
                                       n_bits,
                                       min_value,
                                       max_value,
                                       constrained,
                                       n_iter,
                                       min_threshold=min_threshold,
                                       quant_error_method=qc.QuantizationErrorMethod.NOCLIPPING)


def get_range_selection_tensor_error_function(quant_error_method: qc.QuantizationErrorMethod,
                                              p: int,
                                              norm: bool = False,
                                              n_bits: int = 8) -> Callable:
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for tensor quantization.
    Args:
        quant_error_method: the requested error function type.
        p: p-norm to use for the Lp-norm distance.
        norm: whether to normalize the error function result.
        n_bits: Number of bits to quantize the tensor.


    Returns: a Callable method that calculates the error between a tensor and a quantized tensor.

    """
    quant_method_error_function_mapping = {
        qc.QuantizationErrorMethod.MSE: lambda x, y, mm: compute_mse(x, y, norm=norm),
        qc.QuantizationErrorMethod.MAE: lambda x, y, mm: compute_mae(x, y, norm=norm),
        qc.QuantizationErrorMethod.LP: lambda x, y, mm: compute_lp_norm(x, y, p=p, norm=norm),
        qc.QuantizationErrorMethod.KL: lambda x, y, mm: _kl_error_function(x, range_min=mm[0], range_max=mm[1],
                                                                           n_bits=n_bits)
    }

    return quant_method_error_function_mapping[quant_error_method]


def get_range_selection_histogram_error_function(quant_error_method: qc.QuantizationErrorMethod,
                                                 p: int) -> Callable:
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for histogram quantization.
    Args:
        quant_error_method: the requested error function type.
        p: p-norm to use for the Lp-norm distance.

    Returns: a Callable method that calculates the error between a tensor and a quantized tensor.

    """
    quant_method_error_function_mapping = {
        qc.QuantizationErrorMethod.MSE: lambda q_bins, q_count, bins, counts, threshold, min_max_range:
        _mse_error_histogram(q_bins, q_count, bins, counts),
        qc.QuantizationErrorMethod.MAE: lambda q_bins, q_count, bins, counts, threshold, min_max_range:
        _mae_error_histogram(q_bins, q_count, bins, counts),
        qc.QuantizationErrorMethod.LP: lambda q_bins, q_count, bins, counts, threshold, min_max_range:
        _lp_error_histogram(q_bins, q_count, bins, counts, p=p),
        qc.QuantizationErrorMethod.KL: lambda q_bins, q_count, bins, counts, threshold, min_max_range:
        _kl_error_histogram(q_bins, q_count, bins, counts, min_max_range[0], min_max_range[1]),
    }

    return quant_method_error_function_mapping[quant_error_method]
