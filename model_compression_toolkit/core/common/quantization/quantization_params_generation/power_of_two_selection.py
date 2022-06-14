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

import model_compression_toolkit.core.common.quantization.quantization_config as qc
from model_compression_toolkit.core.common.constants import MIN_THRESHOLD, THRESHOLD
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_search import \
    qparams_selection_tensor_search, qparams_selection_histogram_search
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import power_of_two_constraint
from model_compression_toolkit.core.common.quantization.quantization_params_generation.kl_selection import \
    _kl_error_histogram, _kl_error_function
from model_compression_toolkit.core.common.quantization.quantization_params_generation.error_histograms import \
    _lp_error_histogram, _mae_error_histogram, _mse_error_histogram
from model_compression_toolkit.core.common.quantization.quantization_params_generation.no_clipping import \
    no_clipping_selection_tensor

from model_compression_toolkit.core.common.similarity_analyzer import compute_mse, compute_mae, compute_lp_norm


def power_of_two_selection_tensor(tensor_data: np.ndarray,
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
        Optimal threshold to quantize the tensor in a power of 2 manner.
    """

    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        return no_clipping_selection_tensor(tensor_data, p=p, n_bits=n_bits, min_threshold=min_threshold)
    else:
        signed = np.any(tensor_data < 0)
        error_function = get_threshold_selection_tensor_error_function(quant_error_method, p, norm=False, n_bits=n_bits, signed=signed)
        threshold = qparams_selection_tensor_search(error_function,
                                                    tensor_data,
                                                    n_bits,
                                                    per_channel=per_channel,
                                                    channel_axis=channel_axis,
                                                    n_iter=n_iter,
                                                    min_threshold=min_threshold)
        return {THRESHOLD: threshold}


def power_of_two_selection_histogram(bins: np.ndarray,
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
        Optimal threshold to quantize the histogram a power of 2 manner.
    """
    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        tensor_max = np.max(np.abs(bins))
        threshold = power_of_two_constraint(tensor_max, min_threshold)
    else:
        error_function = get_threshold_selection_histogram_error_function(quant_error_method, p)
        threshold = qparams_selection_histogram_search(error_function,
                                                       bins,
                                                       counts,
                                                       n_bits,
                                                       constrained=constrained,
                                                       n_iter=n_iter,
                                                       min_threshold=min_threshold)
    return {THRESHOLD: threshold}


def get_threshold_selection_tensor_error_function(quant_error_method: qc.QuantizationErrorMethod,
                                                  p: int,
                                                  norm: bool = False,
                                                  n_bits: int = 8,
                                                  signed: bool = True) -> Callable:
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for tensor quantization.
    Args:
        quant_error_method: the requested error function type.
        p: p-norm to use for the Lp-norm distance.
        norm: whether to normalize the error function result.
        n_bits: Number of bits to quantize the tensor.
        signed: signed input

    Returns: a Callable method that calculates the error between a tensor and a quantized tensor.
    """
    quant_method_error_function_mapping = {
        qc.QuantizationErrorMethod.MSE: lambda x, y, threshold: compute_mse(x, y),
        qc.QuantizationErrorMethod.MAE: lambda x, y, threshold: compute_mae(x, y),
        qc.QuantizationErrorMethod.LP: lambda x, y, threshold: compute_lp_norm(x, y, p=p, norm=norm),
        qc.QuantizationErrorMethod.KL: lambda x, y, threshold: _kl_error_function(x, range_min=0 if not signed else -threshold, range_max=threshold, n_bits=n_bits)
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
        qc.QuantizationErrorMethod.KL: lambda q_bins, q_count, bins, counts, threshold, _range:
        _kl_error_histogram(q_bins, q_count, bins, counts, -threshold, threshold)
    }

    return quant_method_error_function_mapping[quant_error_method]
