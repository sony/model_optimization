# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.util.object_identity import Reference as TFReference

from model_compression_toolkit.quantizers_infrastructure.common.constants import EPS, MULTIPLIER_N_BITS


def fix_range_to_include_zero(range_min: np.ndarray,
                              range_max: np.ndarray,
                              n_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjusting the quantization range to include representation of 0.0 in the quantization grid.
    If quantization per-channel, then range_min and range_max should be tensors in the specific shape that allows
    quantization along the channel_axis.

    Args:
        range_min: min bound of the quantization range (before adjustment).
        range_max: max bound of the quantization range (before adjustment).
        n_bits: Number of bits to quantize the tensor.

    Returns: adjusted quantization range
    """
    min_positive = range_min > 0
    max_negative = range_max < 0

    scale = (range_max - range_min) / (2 ** n_bits - 1)

    min_range_adj = scale * np.round(range_min / scale)
    max_range_adj = range_max - range_min + min_range_adj

    mid_range = np.logical_and(np.logical_not(min_positive), np.logical_not(max_negative))
    min_range_adj = min_range_adj * mid_range + max_negative * range_min
    max_range_adj = max_range_adj * mid_range + min_positive * range_max
    return min_range_adj, max_range_adj


def kmeans_assign_clusters(cluster_centers: np.ndarray,
                           query: np.ndarray) -> np.ndarray:
    """
    Assign each data value in query with its closest cluster center point.
    Args:
        cluster_centers: the cluster centers to assign the query values.
        query: values for which to assign cluster centers.

    Returns: A tensor of indexes to the cluster centers that where assigned to each value in
             the query tensor.

    """
    d0 = query.shape[0]
    d1 = cluster_centers.shape[0]
    query_ = query.repeat(d1).reshape(d0, d1)
    cluster_centers_ = cluster_centers.repeat(d0).reshape(d1, d0).transpose(1, 0)
    return np.argmin(np.abs(query_ - cluster_centers_), axis=1)


def lut_kmeans_quantizer(tensor_data: TFReference,
                         cluster_centers: np.ndarray,
                         signed: bool,
                         threshold: np.ndarray) -> TFReference:
    """
    Quantize a tensor using a non-uniform quantization based on the pre-defined kmeans clusters.
    1. Scales tensor_data with the threshold into 8-bit quantization range.
    2. Assigns cluster centers to each value.
    3. Scales back by multiplying the result by threshold and dividing with the quantization range max value.
    The result is the quantized tensor.

    Args:
        tensor_data: Input activation tensor.
        cluster_centers: the cluster centers to assign the tensor values.
        signed: Whether the quantization is signed or not.
        threshold: threshold for quantization.

    Returns: Quantized tensor.
    """

    tensor = int_quantization_with_threshold(tensor_data, n_bits=MULTIPLIER_N_BITS, signed=signed, threshold=threshold)
    tensor = tf.expand_dims(tensor, -1)

    expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
    cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
    centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

    quant_tensor = (centers / (2 ** (MULTIPLIER_N_BITS - int(signed)))) * threshold

    return quant_tensor


def int_quantization_with_threshold(data: TFReference,
                                    n_bits: int,
                                    signed: bool,
                                    threshold: np.ndarray,
                                    eps: float = EPS) -> TFReference:
    """
    Divides data by threshold and quantize it to integers in the quantization range (depends on signed value).

    Args:
        data: tensor data.
        n_bits: number of bits that determines the quantization range.
        signed: Whether the quantization is signed or not.
        threshold: threshold for quantization.
        eps: Small value for numerical stability in division.

    Returns:
        Uniform Quantized tensor.

    """

    if signed:
        clip_max = 2 ** (n_bits - 1) - 1
        clip_min = -2 ** (n_bits - 1)
    else:
        clip_max = 2 ** n_bits - 1
        clip_min = 0

    return tf.clip_by_value((data / (threshold + eps)) * (2 ** (n_bits - int(signed))),
                            clip_value_max=clip_max, clip_value_min=clip_min)
