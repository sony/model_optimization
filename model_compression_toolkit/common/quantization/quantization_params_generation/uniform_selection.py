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
from typing import Callable

import numpy as np

import model_compression_toolkit.common.quantization.quantization_config as qc
from model_compression_toolkit.common.constants import MIN_THRESHOLD, RANGE_MIN, RANGE_MAX
from model_compression_toolkit.common.quantization.quantization_params_generation.kl_selection import \
    _kl_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.lp_selection import \
    _lp_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mae_selection import \
    _mae_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mse_selection import \
    _mse_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_search import \
    uniform_qparams_tensor_minimization, uniform_qparams_histogram_minimization
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import get_tensor_max, \
    get_tensor_min, reshape_tensor_for_per_channel_search


from model_compression_toolkit.common.similarity_analyzer import compute_mse, compute_mae, compute_lp_norm


# TODO: threshold_method is not a good argument to determine the type of RANGE selection method.
#   need to use some different (more robust) argument type to specify methods for parameters selection
def uniform_selection_tensor(tensor_data: np.ndarray,
                             p: int,
                             n_bits: int,
                             per_channel: bool = False,
                             channel_axis: int = 1,
                             n_iter: int = 10,
                             min_threshold: float = MIN_THRESHOLD,
                             threshold_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE) -> dict:
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
        threshold_method: an error function to optimize the range parameters selection accordingly.

    Returns:
        Optimal quantization range to quantize the tensor uniformly.
    """
    tensor_min = get_tensor_min(tensor_data, per_channel, channel_axis)
    tensor_max = get_tensor_max(tensor_data, per_channel, channel_axis)

    if threshold_method == qc.QuantizationErrorMethod.NOCLIPPING:
        return {RANGE_MIN: tensor_min, RANGE_MAX: tensor_max}
    else:
        error_function = get_range_selection_tensor_error_function(threshold_method, p)
        if per_channel:
            tensor_data_r = reshape_tensor_for_per_channel_search(tensor_data, channel_axis)
            output_shape = [-1 if i is channel_axis else 1 for i in range(len(tensor_data.shape))]
            res_min, res_max = [], []

            for j in range(tensor_data_r.shape[0]):  # iterate all channels of the tensor.
                channel_data = tensor_data_r[j, :]
                channel_range_min = tensor_min.flatten()[j]
                channel_range_max = tensor_max.flatten()[j]
                x0 = np.array([channel_range_min, channel_range_max])

                channel_res = uniform_qparams_tensor_minimization(channel_data, x0, n_bits, error_function)

                res_min.append(channel_res.x[0])
                res_max.append(channel_res.x[1])

            res_min = np.reshape(np.array(res_min), output_shape)
            res_max = np.reshape(np.array(res_max), output_shape)
            return {RANGE_MIN: res_min,
                    RANGE_MAX: res_max}
        else:
            x0 = np.array([tensor_min, tensor_max])
            res = uniform_qparams_tensor_minimization(tensor_data, x0, n_bits, error_function)
            return {RANGE_MIN: res.x[0],
                    RANGE_MAX: res.x[1]}


def uniform_selection_histogram(bins: np.ndarray,
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
        threshold_method: an error function to optimize the range parameters selection accordingly.

    Returns:
        Optimal quantization range to quantize the histogram uniformly.
    """
    tensor_min = np.min(bins)
    tensor_max = np.max(bins)
    tensor_min_max = np.array([tensor_min, tensor_max])

    if threshold_method == qc.QuantizationErrorMethod.NOCLIPPING:
        res = tensor_min_max
    else:
        error_function = get_range_selection_histogram_error_function(threshold_method, p)

        # returned 'x' here is an array with min and max range values
        res = uniform_qparams_histogram_minimization(bins, tensor_min_max, n_bits, counts, error_function).x

    return {RANGE_MIN: res[0], RANGE_MAX: res[1]}


# TODO: need to move both "get" methods to a shared location for both uniform and symmetric selectors
#  since they are identical, but can't move to 'quantizers_helpers.py' since it is in a different package and we need
#  to access this package methods (error functions), which causes circulated dependency in the project.
def get_range_selection_tensor_error_function(threshold_method, p):
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


def get_range_selection_histogram_error_function(threshold_method, p):
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
