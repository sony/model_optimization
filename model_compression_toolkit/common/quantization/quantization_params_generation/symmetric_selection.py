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
from model_compression_toolkit.common.quantization.quantization_params_generation.kl_selection import \
    _kl_error_histogram, _kl_error_function
from model_compression_toolkit.common.quantization.quantization_params_generation.lp_selection import \
    _lp_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mae_selection import \
    _mae_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mse_selection import \
    _mse_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_search import \
    symmetric_qparams_histogram_minimization, symmetric_qparams_tensor_minimization
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import get_tensor_max, quantize_tensor,\
    reshape_tensor_for_per_channel_search
from scipy.optimize import minimize

from model_compression_toolkit.common.similarity_analyzer import compute_mse, compute_mae, compute_lp_norm


def symmetric_selection_tensor(tensor_data: np.ndarray,
                               p: int,
                               n_bits: int,
                               per_channel: bool = False,
                               channel_axis: int = 1,
                               n_iter: int = 10,
                               min_threshold: float = MIN_THRESHOLD,
                               threshold_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE) -> dict:
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
        threshold_method: an error function to optimize the threshold selection accordingly.

    Returns:
        Optimal threshold to quantize the tensor in a symmetric manner.
    """

    signed = np.any(tensor_data < 0)  # check if tensor is singed
    unsigned_tensor_data = np.abs(tensor_data)
    tensor_max = get_tensor_max(unsigned_tensor_data, per_channel, channel_axis)

    if threshold_method == qc.QuantizationErrorMethod.NOCLIPPING:
        return {THRESHOLD: tensor_max}
    elif threshold_method == qc.QuantizationErrorMethod.KL:
        # TODO: what about per_channel?
        return {THRESHOLD: minimize(
                fun=lambda threshold: _kl_error_function(quantize_tensor(tensor_data, threshold, n_bits, signed),
                                                         tensor_data, threshold, n_bits=n_bits),
                x0=tensor_max).x}
    else:
        error_function = get_threshold_selection_tensor_error_function(threshold_method, p)
        if per_channel:
            tensor_data_r = reshape_tensor_for_per_channel_search(tensor_data, channel_axis)
            output_shape = [-1 if i is channel_axis else 1 for i in range(len(tensor_data.shape))]
            res = []

            for j in range(tensor_data_r.shape[0]):  # iterate all channels of the tensor.
                channel_data = tensor_data_r[j, :]
                channel_threshold = tensor_max.flatten()[j]
                channel_res = symmetric_qparams_tensor_minimization(channel_data, channel_threshold,
                                                            n_bits, signed, error_function)
                res.append(channel_res.x)
            res = np.reshape(np.array(res), output_shape)
        else:
            # returned 'x' here is the value of the optimal threshold
            res = symmetric_qparams_tensor_minimization(tensor_data, tensor_max, n_bits, signed, error_function)
            res = res.x
        return {THRESHOLD: res}


def symmetric_selection_histogram(bins: np.ndarray,
                                  counts: np.ndarray,
                                  p: int,
                                  n_bits: int,
                                  min_value: float,
                                  max_value: float,
                                  constrained: bool = True,
                                  n_iter: int = 10,
                                  min_threshold: float = MIN_THRESHOLD,
                                  threshold_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE) -> dict:
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
        threshold_method: an error function to optimize the threshold selection accordingly.

    Returns:
        Optimal threshold to quantize the histogram a symmetric manner.
    """
    tensor_max = np.max(np.abs(bins))
    signed = np.any(bins < 0)  # check if tensor is singed
    if threshold_method == qc.QuantizationErrorMethod.NOCLIPPING:
        return {THRESHOLD: tensor_max}
    else:
        error_function = get_threshold_selection_histogram_error_function(threshold_method, p)
        res = symmetric_qparams_histogram_minimization(bins, tensor_max, n_bits, signed, counts, error_function)
        return {THRESHOLD: res.x[0]}


def get_threshold_selection_tensor_error_function(threshold_method, p):
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for tensor quantization.
    Args:
        threshold_method: the requested error function type.
        p: p-norm to use for the Lp-norm distance.


    Returns: a Callable method that calculates the error between a tensor and a quantized tensor.

    """
    threshold_method_error_function_mapping = {
        qc.QuantizationErrorMethod.MSE: compute_mse,
        qc.QuantizationErrorMethod.MAE: compute_mae,
        qc.QuantizationErrorMethod.LP: lambda x, y: compute_lp_norm(x, y, p),
    }

    return threshold_method_error_function_mapping[threshold_method]


def get_threshold_selection_histogram_error_function(threshold_method, p):
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for histogram quantization.
    Args:
        threshold_method: the requested error function type.
        p: p-norm to use for the Lp-norm distance.

    Returns: a Callable method that calculates the error between a tensor and a quantized tensor.

    """
    threshold_method_error_function_mapping = {
        qc.QuantizationErrorMethod.MSE: _mse_error_histogram,
        qc.QuantizationErrorMethod.MAE: _mae_error_histogram,
        qc.QuantizationErrorMethod.LP: lambda q_bins, q_count, bins, counts:
            _lp_error_histogram(q_bins, q_count, bins, counts, p=p),
        qc.QuantizationErrorMethod.KL: _kl_error_histogram
    }

    return threshold_method_error_function_mapping[threshold_method]
