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
from model_compression_toolkit.common.constants import MIN_THRESHOLD, THRESHOLD
from model_compression_toolkit.common.quantization.quantization_params_generation.kl_selection import \
    _kl_error_histogram, _kl_error_function
from model_compression_toolkit.common.quantization.quantization_params_generation.lp_selection import \
    _lp_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mae_selection import \
    _mae_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mse_selection import \
    _mse_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_search import \
    qparams_symmetric_selection_tensor_search, \
    qparams_symmetric_selection_histogram_search, kl_qparams_symmetric_selection_histogram_search
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import \
    get_tensor_max

from model_compression_toolkit.common.similarity_analyzer import compute_mse, compute_mae, compute_lp_norm


def symmetric_selection_tensor(tensor_data: np.ndarray,
                               p: int,
                               n_bits: int,
                               per_channel: bool = False,
                               channel_axis: int = 1,
                               n_iter: int = 10,
                               min_threshold: float = MIN_THRESHOLD,
                               quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE) -> dict:
    """
    Compute the optimal threshold based on the provided QuantizationErrorMethod to quantize the tensor.
    Different search is applied, depends on the value of the selected QuantizationErrorMethod.

    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_iter: Number of iterations to search for the optimal threshold (not used for this method).
        min_threshold: Minimal threshold to use if threshold is too small (not used for this method).
        quant_error_method: an error function to optimize the parameters' selection accordingly.

    Returns:
        Optimal threshold to quantize the tensor in a symmetric manner.
    """

    unsigned_tensor_data = np.abs(tensor_data)
    tensor_max = get_tensor_max(unsigned_tensor_data, per_channel, channel_axis)

    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        threshold = get_init_threshold(min_threshold, tensor_max, per_channel)
    else:
        error_function = get_threshold_selection_tensor_error_function(quant_error_method, p, norm=False, n_bits=n_bits)
        threshold = qparams_symmetric_selection_tensor_search(error_function,
                                                              tensor_data,
                                                              tensor_max,
                                                              n_bits,
                                                              per_channel,
                                                              channel_axis,
                                                              min_threshold=min_threshold)
    return {THRESHOLD: threshold}


def symmetric_selection_histogram(bins: np.ndarray,
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
    Compute the optimal threshold based on the provided QuantizationErrorMethod to quantize a histogram.
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
        min_threshold: Minimal threshold to use if threshold is too small (used only for kl threshold selection).
        quant_error_method: an error function to optimize the parameters' selection accordingly.

    Returns:
        Optimal threshold to quantize the histogram a symmetric manner.
    """
    tensor_max = np.max(np.abs(bins)[1:][counts > 0])
    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        threshold = get_init_threshold(min_threshold, tensor_max)
    elif quant_error_method == qc.QuantizationErrorMethod.KL:
        # search for KL error is separated because the error method signature is different from the other error methods.
        threshold = kl_qparams_symmetric_selection_histogram_search(_kl_error_histogram,
                                                                    tensor_max,
                                                                    bins,
                                                                    counts,
                                                                    n_bits,
                                                                    min_threshold=min_threshold)
    else:
        error_function = get_threshold_selection_histogram_error_function(quant_error_method, p)
        threshold = qparams_symmetric_selection_histogram_search(error_function,
                                                                 tensor_max,
                                                                 bins,
                                                                 counts,
                                                                 n_bits,
                                                                 min_threshold=min_threshold)
    return {THRESHOLD: threshold}


def symmetric_no_clipping_selection_min_max(bins: np.ndarray,
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
    Gets a threshold between min and max numbers.
    If computed threshold is less than min_threshold, min_threshold is returned.

    Returns:
        A constrained threshold of the min/max values.
    """
    return symmetric_selection_histogram(np.asarray([min_value, max_value]),  # histogram with min-max values only
                                         np.asarray([1]),  # passing dummy counts just to make the dimensions work
                                         p,
                                         n_bits,
                                         min_value,
                                         max_value,
                                         constrained,
                                         n_iter,
                                         min_threshold=min_threshold,
                                         quant_error_method=qc.QuantizationErrorMethod.NOCLIPPING)


def get_threshold_selection_tensor_error_function(quant_error_method: qc.QuantizationErrorMethod,
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
        qc.QuantizationErrorMethod.MSE: lambda x, y, t: compute_mse(x, y, norm=norm),
        qc.QuantizationErrorMethod.MAE: lambda x, y, t: compute_mae(x, y, norm=norm),
        qc.QuantizationErrorMethod.LP: lambda x, y, t: compute_lp_norm(x, y, p=p, norm=norm),
        qc.QuantizationErrorMethod.KL: lambda x, y, t: _kl_error_function(x, range_min=-t, range_max=t, n_bits=n_bits)
    }

    return quant_method_error_function_mapping[quant_error_method]


def get_threshold_selection_histogram_error_function(quant_error_method: qc.QuantizationErrorMethod,
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
        qc.QuantizationErrorMethod.MSE: lambda q_bins, q_count, bins, counts, threshold, _range:
        _mse_error_histogram(q_bins, q_count, bins, counts),
        qc.QuantizationErrorMethod.MAE: lambda q_bins, q_count, bins, counts, threshold, _range:
        _mae_error_histogram(q_bins, q_count, bins, counts),
        qc.QuantizationErrorMethod.LP: lambda q_bins, q_count, bins, counts, threshold, _range:
        _lp_error_histogram(q_bins, q_count, bins, counts, p=p),
    }

    return quant_method_error_function_mapping[quant_error_method]


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
