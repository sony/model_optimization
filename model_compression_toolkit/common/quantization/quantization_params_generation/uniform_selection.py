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
import numpy as np
from scipy import optimize

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
    qparams_histogram_minimization, kl_uniform_qparams_histogram_minimization, \
    uniform_qparams_selection_per_channel_search, qparams_tensor_minimization
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import get_tensor_max, \
    get_tensor_min, uniform_quantize_tensor

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
        res = tensor_min, tensor_max
    elif quant_error_method == qc.QuantizationErrorMethod.KL:
        # search for KL error is separated because the error method signature is different from the other error methods.
        # we use _kl_batch_error_function to allow calculation per_channel in a vectorized manner if necessary,
        # we pass it as argument to avoid exposing protected package member inside kl_uniform_quantization_loss.
        if per_channel:
            # Using search per-channel wrapper for kl based minimization
            res = uniform_qparams_selection_per_channel_search(tensor_data, tensor_min, tensor_max, channel_axis,
                                                               search_function=lambda _x, _x0:
                                                               optimize.minimize(fun=lambda min_max_range: _kl_error_function(_x,
                                                                                                                              range_min=min_max_range[0],
                                                                                                                              range_max=min_max_range[1],
                                                                                                                              n_bits=n_bits),
                                                                                 x0=_x0)
                                                               )
        else:
            x0 = np.array([tensor_min, tensor_max])
            res = optimize.minimize(fun=lambda min_max_range: _kl_error_function(tensor_data,
                                                                                 range_min=min_max_range[0],
                                                                                 range_max=min_max_range[1],
                                                                                 n_bits=n_bits),
                                    x0=x0)
            # returned 'x' here is an array with min and max range values
            res = res.x[0], res.x[1]
    else:
        error_function = get_range_selection_tensor_error_function(quant_error_method, p)
        if per_channel:
            # Using search per-channel wrapper for minimization
            res = uniform_qparams_selection_per_channel_search(tensor_data, tensor_min, tensor_max, channel_axis,
                                                               search_function=lambda _x, _x0:
                                                               qparams_tensor_minimization(_x, _x0, error_function,
                                                                                           quant_function=lambda min_max_range:
                                                                                           uniform_quantize_tensor(_x,
                                                                                                                   min_max_range[0],
                                                                                                                   min_max_range[1],
                                                                                                                   n_bits))
                                                               )
        else:
            x0 = np.array([tensor_min, tensor_max])
            res = qparams_tensor_minimization(tensor_data, x0, error_function,
                                              quant_function=lambda min_max_range: uniform_quantize_tensor(tensor_data,
                                                                                                           min_max_range[0],
                                                                                                           min_max_range[1],
                                                                                                           n_bits)
                                              )
            # returned 'x' here is an array with min and max range values
            res = res.x[0], res.x[1]

    return {RANGE_MIN: res[0],
            RANGE_MAX: res[1]}


def uniform_selection_histogram(bins: np.ndarray,
                                counts: np.ndarray,
                                p: int,
                                n_bits: int,
                                min_value: float,
                                max_value: float,
                                constrained: bool = True,
                                n_iter: int = 10,
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
    tensor_min = np.min(bins)
    tensor_max = np.max(bins)
    tensor_min_max = np.array([tensor_min, tensor_max])

    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        res = tensor_min_max
    elif quant_error_method == qc.QuantizationErrorMethod.KL:
        # search for KL error is separated because the error method signature is different from the other error methods.
        # we pass it as argument to avoid exposing protected package member inside kl_uniform_quantization_loss.
        error_function = _kl_error_histogram
        res = kl_uniform_qparams_histogram_minimization(bins,
                                                        tensor_min_max,
                                                        counts,
                                                        n_bits,
                                                        error_function)

        res = res.x

    else:
        error_function = get_range_selection_histogram_error_function(quant_error_method, p)
        res = qparams_histogram_minimization(bins,
                                             tensor_min_max,
                                             counts,
                                             error_function,
                                             quant_function=lambda min_max_range:
                                             uniform_quantize_tensor(bins, min_max_range[0], min_max_range[1], n_bits))
        # returned 'x' here is an array with min and max range values
        res = res.x

    return {RANGE_MIN: res[0], RANGE_MAX: res[1]}


def get_range_selection_tensor_error_function(quant_error_method, p):
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for tensor quantization.
    Args:
        quant_error_method: the requested error function type.
        p: p-norm to use for the Lp-norm distance.


    Returns: a Callable method that calculates the error between a tensor and a quantized tensor.

    """
    quant_method_error_function_mapping = {
        qc.QuantizationErrorMethod.MSE: compute_mse,
        qc.QuantizationErrorMethod.MAE: compute_mae,
        qc.QuantizationErrorMethod.LP: lambda x, y: compute_lp_norm(x, y, p),
    }

    return quant_method_error_function_mapping[quant_error_method]


def get_range_selection_histogram_error_function(quant_error_method, p):
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for histogram quantization.
    Args:
        quant_error_method: the requested error function type.
        p: p-norm to use for the Lp-norm distance.

    Returns: a Callable method that calculates the error between a tensor and a quantized tensor.

    """
    quant_method_error_function_mapping = {
        qc.QuantizationErrorMethod.MSE: lambda q_bins, q_count, bins, counts, threshold:
            _mse_error_histogram(q_bins, q_count, bins, counts),
        qc.QuantizationErrorMethod.MAE: lambda q_bins, q_count, bins, counts, threshold:
            _mae_error_histogram(q_bins, q_count, bins, counts),
        qc.QuantizationErrorMethod.LP: lambda q_bins, q_count, bins, counts, threshold:
            _lp_error_histogram(q_bins, q_count, bins, counts, p=p),
    }

    return quant_method_error_function_mapping[quant_error_method]
