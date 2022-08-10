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

import numpy as np
from sklearn.cluster import KMeans

import model_compression_toolkit.core.common.quantization.quantization_config as qc
from model_compression_toolkit.core.common.constants import CLUSTER_CENTERS, SCALE_PER_CHANNEL, MIN_THRESHOLD, EPS


def kmeans_tensor(tensor_data: np.ndarray,
                  p: int,
                  n_bits: int,
                  per_channel: bool = False,
                  channel_axis: int = 1,
                  n_iter: int = 10,
                  min_threshold: float = MIN_THRESHOLD,
                  quant_error_method: qc.QuantizationErrorMethod = None) -> dict:
    """
    Compute the 2^nbit cluster assignments for the given tensor according to the k-means algorithm.

    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_iter: Number of iterations to search_methods for the optimal threshold.
        min_threshold: Minimal threshold to chose when the computed one is smaller.
        quant_error_method: an error function to optimize the parameters' selection accordingly (not used for this method).

    Returns:
        A dictionary containing the cluster assignments according to the k-means algorithm and the scales per channel.
    """
    if len(np.unique(tensor_data.flatten())) < 2 ** n_bits:
        n_clusters = len(np.unique(tensor_data.flatten()))
    else:
        n_clusters = 2 ** n_bits
    kmeans = KMeans(n_clusters=n_clusters)
    axis_not_channel = [i for i in range(len(tensor_data.shape))]
    if channel_axis in axis_not_channel:
        axis_not_channel.remove(channel_axis)
    if per_channel:
        scales_per_channel = np.max(np.abs(tensor_data), axis=tuple(axis_not_channel), keepdims=True)
    else:
        scales_per_channel = np.max(np.abs(tensor_data), keepdims=True)
    tensor_for_kmeans = (tensor_data / (scales_per_channel + EPS))
    kmeans.fit(tensor_for_kmeans.reshape(-1, 1))

    return {CLUSTER_CENTERS: kmeans.cluster_centers_,
            SCALE_PER_CHANNEL: scales_per_channel,
            }