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
from model_compression_toolkit.core.common.quantization.quantization_params_generation.kl_selection import \
    _kl_error_histogram, _kl_error_function
from model_compression_toolkit.core.common.similarity_analyzer import compute_mse, compute_mae, compute_lp_norm
from model_compression_toolkit.core.common.target_platform import QuantizationMethod


def _mse_error_histogram(q_bins: np.ndarray,
                         q_count: np.ndarray,
                         bins: np.ndarray,
                         counts: np.ndarray) -> np.float:
    """
    Compute the error function between a histogram to its quantized version.
    The error is computed based on the mean square error the distributions have.

    Args:
        q_bins: Bins values of the quantized histogram.
        q_count: Bins counts of the quantized histogram.
        bins: Bins values of the original histogram.
        counts: Bins counts of the original histogram.

    Returns:
        MSE between the two histograms.
    """

    return np.sum((np.power((q_bins - bins)[:-1], 2.0) * counts)) / np.sum(counts)


def _mae_error_histogram(q_bins: np.ndarray,
                         q_count: np.ndarray,
                         bins: np.ndarray,
                         counts: np.ndarray) -> np.ndarray:
    """
    Compute the error function between a histogram to its quantized version.
    The error is computed using the mean absolute error between the two histograms.

    Args:
        q_bins: Bins values of the quantized histogram.
        q_count: Bins counts of the quantized histogram.
        bins: Bins values of the original histogram.
        counts: Bins counts of the original histogram.

    Returns:
        Mean absolute error of the two histograms.
    """

    return np.sum((np.abs((q_bins - bins)[:-1]) * counts)) / np.sum(counts)


def _lp_error_histogram(q_bins: np.ndarray,
                        q_count: np.ndarray,
                        bins: np.ndarray,
                        counts: np.ndarray,
                        p: int) -> np.float:
    """
    Compute the error function between a histogram to its quantized version.
    The error is computed based on the distance in Lp-norm between the two distributions.
    The p-norm to use for the distance computing is passed.

    Args:
        q_bins: Bins values of the quantized histogram.
        q_count: Bins counts of the quantized histogram.
        bins: Bins values of the original histogram.
        counts: Bins counts of the original histogram.
        p: p-norm to use for the Lp-norm distance.

    Returns:
        The Lp-norm distance between the two histograms.
    """

    return np.sum((np.power(np.abs((q_bins - bins)[:-1]), p) * counts)) / np.sum(counts)


def get_threshold_selection_tensor_error_function(quantization_method: QuantizationMethod,
                                                  quant_error_method: qc.QuantizationErrorMethod,
                                                  p: int,
                                                  norm: bool = False,
                                                  n_bits: int = 8,
                                                  signed: bool = True) -> Callable:
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for tensor quantization.
    Args:
        quantization_method: Quantization method for threshold selection
        quant_error_method: the requested error function type.
        p: p-norm to use for the Lp-norm distance.
        norm: whether to normalize the error function result.
        n_bits: Number of bits to quantize the tensor.
        signed: signed input

    Returns: a Callable method that calculates the error between a tensor and a quantized tensor.
    """

    quant_method_error_function_mapping = {
        qc.QuantizationErrorMethod.MSE: lambda x, y, threshold: compute_mse(x, y, norm=norm),
        qc.QuantizationErrorMethod.MAE: lambda x, y, threshold: compute_mae(x, y, norm=norm),
        qc.QuantizationErrorMethod.LP: lambda x, y, threshold: compute_lp_norm(x, y, p=p, norm=norm),
        qc.QuantizationErrorMethod.KL:
            lambda x, y, threshold: _kl_error_function(x, range_min=threshold[0], range_max=threshold[1],
                                                       n_bits=n_bits) if quantization_method == QuantizationMethod.UNIFORM
            else _kl_error_function(x, range_min=0 if not signed else -threshold, range_max=threshold, n_bits=n_bits)
    }

    return quant_method_error_function_mapping[quant_error_method]


def get_threshold_selection_histogram_error_function(quantization_method: QuantizationMethod,
                                                     quant_error_method: qc.QuantizationErrorMethod,
                                                     p: int) -> Callable:
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for histogram quantization.
    Args:
        quantization_method: Quantization method for threshold selection
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
        qc.QuantizationErrorMethod.KL:
            lambda q_bins, q_count, bins, counts, threshold, _range: _kl_error_histogram(q_bins, q_count, bins, counts, _range[0], _range[1])
            if quantization_method == QuantizationMethod.UNIFORM
            else _kl_error_histogram(q_bins, q_count, bins, counts, -threshold, threshold)
    }

    return quant_method_error_function_mapping[quant_error_method]
