# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Union, Tuple, Dict

import model_compression_toolkit.core.common.quantization.quantization_config as qc
from model_compression_toolkit.constants import MIN_THRESHOLD, RANGE_MIN, RANGE_MAX, NUM_QPARAM_HESSIAN_SAMPLES, SIGNED
from model_compression_toolkit.core.common.hessian import HessianInfoService
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_search import \
    qparams_uniform_selection_tensor_search, qparams_uniform_selection_histogram_search
from model_compression_toolkit.core.common.quantization.quantization_params_generation.error_functions import \
    get_threshold_selection_tensor_error_function, get_threshold_selection_histogram_error_function
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import get_tensor_max, \
    get_tensor_min
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.common.similarity_analyzer import compute_mse
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import uniform_quantize_tensor


def uniform_selection_tensor(tensor_data: np.ndarray,
                             p: int,
                             n_bits: int,
                             per_channel: bool = False,
                             channel_axis: int = 1,
                             n_iter: int = 10,
                             min_threshold: float = MIN_THRESHOLD,
                             quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE,
                             node=None,
                             hessian_info_service: HessianInfoService = None,
                             num_hessian_samples: int = NUM_QPARAM_HESSIAN_SAMPLES,
                             ) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Compute the optimal quantization range based on the provided QuantizationErrorMethod
    to uniformly quantize the tensor.
    Different search is applied, depends on the value of the selected QuantizationErrorMethod.

    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index. if None, search for best axis.
        n_iter: Number of iterations to search for the optimal threshold (not used for this method).
        min_threshold: Minimal threshold to use if threshold is too small (not used for this method).
        quant_error_method: an error function to optimize the range parameters' selection accordingly.
        node: The node for which the quantization error is computed (used only with HMSE error method).
        hessian_info_service: HessianInfoService object for retrieving Hessian-based scores (used only with HMSE error method).
        num_hessian_samples: Number of samples to approximate Hessian-based scores on (used only with HMSE error method).

    Returns:
        Optimal quantization range to quantize the tensor uniformly.
        Selected quantization channel axis.
    """
    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        if channel_axis is None and per_channel:
            total_error_list = []
            th_list = []
            for _axis in range(len(tensor_data.shape)):
                tensor_min = get_tensor_min(tensor_data, per_channel, _axis)
                tensor_max = get_tensor_max(tensor_data, per_channel, _axis, n_bits, is_uniform_quantization=True)
                q_tensor_data = uniform_quantize_tensor(tensor_data, tensor_min, tensor_max, n_bits)
                total_error_list.append(compute_mse(tensor_data, q_tensor_data, norm=True))
                th_list.append((tensor_min, tensor_max))
            channel_axis = np.argmin(total_error_list)
            mm = th_list[channel_axis]
        else:
            tensor_min = get_tensor_min(tensor_data, per_channel, channel_axis)
            tensor_max = get_tensor_max(tensor_data, per_channel, channel_axis, n_bits, is_uniform_quantization=True)
            mm = tensor_min, tensor_max
    else:
        axis = -1 if per_channel else None
        error_function = get_threshold_selection_tensor_error_function(QuantizationMethod.UNIFORM, quant_error_method,
                                                                       p, axis=axis, norm=False, node=node,
                                                                       hessian_info_service=hessian_info_service,
                                                                       num_hessian_samples=num_hessian_samples)
        mm, channel_axis = qparams_uniform_selection_tensor_search(error_function,
                                                                   tensor_data,
                                                                   n_bits,
                                                                   per_channel,
                                                                   channel_axis)
    # In case the tensor\axis has a single value, then min==max, so need to adjust either min or max to zero.
    if not isinstance(mm[0], np.ndarray):
        if mm[0] > 0:
            mm = (np.float32(0).astype(mm[0].dtype), mm[1])
        if mm[1] < 0:
            mm = (mm[0], np.float32(0).astype(mm[1].dtype))
    else:
        adj_min_to_zero = np.logical_and(mm[1] == mm[0], mm[0] > 0)
        adj_max_to_zero = np.logical_and(mm[1] == mm[0], mm[1] < 0)
        mm[0][adj_min_to_zero] = 0
        mm[1][adj_max_to_zero] = 0
    return {RANGE_MIN: mm[0],
            RANGE_MAX: mm[1]}, channel_axis


def uniform_selection_histogram(bins: np.ndarray,
                                counts: np.ndarray,
                                p: int,
                                n_bits: int,
                                min_value: float,
                                max_value: float,
                                constrained: bool = True,
                                n_iter: int = 20,
                                min_threshold: float = MIN_THRESHOLD,
                                quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE,
                                is_signed: bool = None) -> Dict:
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
        is_signed: Whether the quantization is signed or not. If None then compute SIGNED value.

    Returns:
        Optimal quantization range to quantize the histogram uniformly.
    """
    tensor_min = np.min(bins[:-1][counts > 0])
    tensor_max = np.max(bins[1:][counts > 0])
    tensor_min_max = np.array([tensor_min, tensor_max])

    signed = tensor_min < 0 if is_signed is None else is_signed
    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        mm = tensor_min_max
    else:
        error_function = get_threshold_selection_histogram_error_function(QuantizationMethod.UNIFORM, quant_error_method, p)
        mm = qparams_uniform_selection_histogram_search(error_function,
                                                        tensor_min_max,
                                                        bins,
                                                        counts,
                                                        n_bits)

    return {RANGE_MIN: mm[0],
            RANGE_MAX: mm[1], SIGNED: signed}


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
                                          qc.QuantizationErrorMethod.NOCLIPPING,
                                          is_signed: bool = None) -> Dict:
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
                                       quant_error_method=qc.QuantizationErrorMethod.NOCLIPPING,
                                       is_signed=is_signed)
