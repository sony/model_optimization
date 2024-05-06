# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Dict
import numpy as np
from sklearn.cluster import KMeans

import model_compression_toolkit.core.common.quantization.quantization_config as qc
from model_compression_toolkit.constants import LUT_VALUES, MIN_THRESHOLD, SCALE_PER_CHANNEL, \
    LUT_VALUES_BITWIDTH, THRESHOLD, NUM_QPARAM_HESSIAN_SAMPLES
from model_compression_toolkit.core.common.hessian import HessianInfoService
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import \
    max_power_of_two, int_quantization_with_threshold
from model_compression_toolkit.core.common.quantization.quantization_params_generation.symmetric_selection import \
    symmetric_selection_tensor
from model_compression_toolkit.core.common.quantization.quantization_params_generation.power_of_two_selection import \
    power_of_two_selection_tensor

from model_compression_toolkit.logger import Logger


def lut_kmeans_tensor(tensor_data: np.ndarray,
                      p: int,
                      n_bits: int,
                      per_channel: bool = False,
                      channel_axis: int = 1,
                      n_iter: int = 10,
                      min_threshold: float = MIN_THRESHOLD,
                      quant_error_method: qc.QuantizationErrorMethod = None,
                      is_symmetric: bool = False,
                      node=None,
                      hessian_info_service: HessianInfoService = None,
                      num_hessian_samples: int = NUM_QPARAM_HESSIAN_SAMPLES) -> Dict:
    """
    The quantizer first finds the closest max value per channel of tensor_data.
    Now, we divide tensor_data with the threshold vector per channel. In addition, we scale the result to the range
    [-2^(LUT_VALUES_BITWIDTH-1), 2^(LUT_VALUES_BITWIDTH-1)-1].
    Next, we take the scaled tensor_data and perform k-means clustering with 2^nbit clusters.
    We return the rounded cluster centers, and threshold per channel. We use these to quantize the data.
    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_iter: Number of iterations to search_methods for the optimal threshold.
        min_threshold: Minimal threshold to chose when the computed one is smaller.
        quant_error_method: an error function to optimize the parameters' selection accordingly (not used for this method).
        is_symmetric (bool): Whether to apply symmetric weight quantization (default is False, meaning power of 2 quantization).
        node: The node for which the quantization error is computed (not used for this method).
        hessian_info_service: HessianInfoService object for retrieving Hessian-based scores (not used for this method).
        num_hessian_samples: Number of samples to approximate Hessian-based scores on (not used for this method).

    Returns:
        A dictionary containing the cluster assignments according to the k-means algorithm,
        the thresholds per channel and the multiplier num bits.
    """
    if n_bits >= LUT_VALUES_BITWIDTH:
        Logger.critical(f'Look-Up-Table (LUT) bit configuration exceeds maximum: {n_bits} bits provided, must be less than {LUT_VALUES_BITWIDTH} bits.')  # pragma: no cover
    # TODO: need to set this externally
    if len(np.unique(tensor_data.flatten())) < 2 ** n_bits:
        n_clusters = len(np.unique(tensor_data.flatten()))
    else:
        n_clusters = 2 ** n_bits
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)

    threshold_selection_tensor = symmetric_selection_tensor if is_symmetric else power_of_two_selection_tensor
    thresholds_per_channel = threshold_selection_tensor(tensor_data, p, n_bits, per_channel,
                                                        channel_axis, n_iter, min_threshold,
                                                        qc.QuantizationErrorMethod.NOCLIPPING)[THRESHOLD]

    tensor_for_kmeans = int_quantization_with_threshold(tensor_data, thresholds_per_channel, LUT_VALUES_BITWIDTH)
    kmeans.fit(tensor_for_kmeans.reshape(-1, 1))

    # Add 0 to the LUT
    cc = np.round(kmeans.cluster_centers_)
    closest2zero_idx = (np.abs(cc - 0)).argmin()
    cc[closest2zero_idx] = 0.0

    return {LUT_VALUES: cc,
            SCALE_PER_CHANNEL: thresholds_per_channel}


def lut_kmeans_histogram(bins: np.ndarray,
                         counts: np.ndarray,
                         p: int,
                         n_bits: int,
                         min_value: float,
                         max_value: float,
                         constrained: bool = True,
                         n_iter: int = 20,
                         min_threshold: float = MIN_THRESHOLD,
                         quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE) -> Dict:
    """
    Finds quantization cluster points for non-uniform activation quantization.
    The quantizer first finds the closest power-of-two number to the max value of the given histogram,
    and scales the bins within 8-bit quantization range.
    Next, it performs a weighted k-means clustering with 2^nbit clusters (using the histogram counts as weights).
    Returns the rounded cluster centers, and 8-bit quantization threshold.

    Args:
        bins: Bins values of the histogram.
        counts: Bins counts of the histogram.
        p: p-norm to use for the Lp-norm distance (not used for this method).
        n_bits: Number of bits to quantize the tensor.
        min_value: Min value (not used for this method).
        max_value: Max value (not used for this method).
        constrained: Whether the threshold should be constrained or not (not used for this method).
        n_iter: Number of iteration ot search for the threshold (not used for this method).
        min_threshold: Minimal threshold to use if threshold is too small.
        quant_error_method: an error function to optimize the parameters' selection accordingly (not used for this method).

    Returns:
        A dictionary containing the cluster assignments according to the k-means algorithm and
        the threshold for pre-clustering quantization.
    """

    if n_bits >= LUT_VALUES_BITWIDTH:
        Logger.critical(f'Look-Up-Table (LUT) bit configuration exceeds maximum: {n_bits} bits provided, must be less than {LUT_VALUES_BITWIDTH} bits.')  # pragma: no cover

    bins_with_values = np.abs(bins)[1:][counts > 0]
    if len(np.unique(bins_with_values.flatten())) < 2 ** n_bits:
        n_clusters = len(np.unique(bins_with_values.flatten()))
    else:
        n_clusters = 2 ** n_bits

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    tensor_max = np.max(bins_with_values)
    threshold = max_power_of_two(tensor_max, min_threshold)

    signed = np.any(bins[:-1][counts != 0] < 0)  # Whether histogram contains negative values or not.
    tensor_for_kmeans = int_quantization_with_threshold(data=bins, threshold=threshold, n_bits=LUT_VALUES_BITWIDTH, signed=signed)
    kmeans.fit(tensor_for_kmeans.reshape(-1, 1), sample_weight=np.insert(counts, 0, 0))

    return {LUT_VALUES: np.float32(np.round(kmeans.cluster_centers_)),
            THRESHOLD: threshold}
