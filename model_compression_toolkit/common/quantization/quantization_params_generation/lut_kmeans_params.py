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

from typing import List

import numpy as np
from sklearn.cluster import KMeans

from model_compression_toolkit.common.constants import CLUSTER_CENTERS, MIN_THRESHOLD, SCALE_PER_CHANNEL, \
    MULTIPLIER_N_BITS
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import kmeans_assign_clusters, \
    int_quantization_with_scale


def lut_kmeans_tensor(tensor_data: np.ndarray,
                      p: int,
                      n_bits: int,
                      per_channel: bool = False,
                      channel_axis: int = 1,
                      n_iter: int = 10,
                      min_threshold: float = MIN_THRESHOLD) -> dict:
    """
    The quantizer first finds the closest power-of-two number to the max value per channel of tensor_data.
    Now, we divide tensor_data with the scale vector per channel. In addition, we scale the result to the range
    [-2^(MULTIPLIER_N_BITS-1), 2^(MULTIPLIER_N_BITS-1)-1].
    Next, we take the scaled tensor_data and perform k-means clustering with 2^nbit clusters.
    We return the rounded cluster centers, and scale per channel. We use these to quantize the data.
    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_iter: Number of iterations to search_methods for the optimal threshold.
        min_threshold: Minimal threshold to chose when the computed one is smaller.

    Returns:
        A dictionary containing the cluster assignments according to the k-means algorithm,
        the scales per channel and the multiplier num bits.
    """
    if n_bits > MULTIPLIER_N_BITS:
        raise Exception(f'Look-Up-Table bit configuration has {n_bits} bits. It must be less or equal to {MULTIPLIER_N_BITS}')
    # TODO: need to set this externally
    if len(np.unique(tensor_data.flatten())) < 2 ** n_bits:
        n_clusters = len(np.unique(tensor_data.flatten()))
    else:
        n_clusters = 2 ** n_bits
    kmeans = KMeans(n_clusters=n_clusters)
    axis_not_channel = [i for i in range(len(tensor_data.shape))]
    axis_not_channel.remove(channel_axis)
    if per_channel:
        scales_per_channel = np.max(np.abs(tensor_data), axis=tuple(axis_not_channel), keepdims=True)
    else:
        scales_per_channel = np.max(np.abs(tensor_data), keepdims=True)
    scales_per_channel = np.power(2.0, np.ceil(np.log2(scales_per_channel)))
    tensor_for_kmeans = int_quantization_with_scale(tensor_data, scales_per_channel, MULTIPLIER_N_BITS)
    kmeans.fit(tensor_for_kmeans.reshape(-1, 1))

    return {CLUSTER_CENTERS: np.round(kmeans.cluster_centers_),
            SCALE_PER_CHANNEL: scales_per_channel,
            }