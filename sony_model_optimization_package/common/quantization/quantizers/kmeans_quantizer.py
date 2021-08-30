# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================

from sklearn.cluster import KMeans
import numpy as np

from sony_model_optimization_package.common.constants import CLUSTER_CENTERS, MIN_THRESHOLD, SCALE_PER_CHANNEL
from sony_model_optimization_package.common.quantization.quantizers.quantizers_helpers import kmeans_assign_clusters


def kmeans_quantizer(tensor_data: np.ndarray,
                        n_bits: int,
                        signed: bool,
                        quantization_params: dict,
                        per_channel: bool,
                        output_channels_axis: int) -> np.ndarray:
    """
    Quantize a tensor according to k-means algorithm. This function assigns cluster centers
    to the tensor data values.

    Args:
        tensor_data: Tensor values to quantize.
        n_bits: Number of bits to quantize the tensor.
        signed: Whether the tensor contains negative values or not.
        quantization_params: Dictionary of specific parameters for this quantization function.
        per_channel: Whether to use separate quantization per output channel.
        output_channels_axis: Axis of the output channel.

    Returns:
        Quantized data.
    """
    cluster_centers = quantization_params[CLUSTER_CENTERS]
    scales_per_channel = quantization_params[SCALE_PER_CHANNEL]
    tensor = (tensor_data / scales_per_channel)
    shape_before_kmeans = tensor.shape
    cluster_assignments = kmeans_assign_clusters(cluster_centers, tensor.reshape(-1, 1))
    quant_tensor = cluster_centers[cluster_assignments].reshape(shape_before_kmeans)
    if per_channel:
        quant_tensor = (quant_tensor * scales_per_channel)
    return quant_tensor


def kmeans_tensor(tensor_data: np.ndarray,
                 p: int,
                 n_bits: int,
                 per_channel: bool = False,
                 channel_axis: int = 1,
                 n_iter: int = 10,
                 min_threshold: float = MIN_THRESHOLD) -> dict:
    """
    Compute the 2^nbit cluster assignments for the given tensor according to the k-means algorithm.

    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_iter: Number of iterations to search for the optimal threshold.
        min_threshold: Minimal threshold to chose when the computed one is smaller.

    Returns:
        A dictionary containing the cluster assignments according to the k-means algorithm and the scales per channel.
    """
    kmeans = KMeans(n_clusters=2 ** n_bits)
    axis_not_channel = [i for i in range(len(tensor_data.shape))]
    if channel_axis in axis_not_channel:
        axis_not_channel.remove(channel_axis)
    scales_per_channel = np.max(np.abs(tensor_data), axis=tuple(axis_not_channel), keepdims=True)
    if per_channel:
        tensor_for_kmeans = (tensor_data / scales_per_channel)
    else:
        scales_per_channel = np.ones_like(scales_per_channel)
        tensor_for_kmeans = tensor_data

    kmeans.fit(tensor_for_kmeans.reshape(-1, 1))

    return {CLUSTER_CENTERS: kmeans.cluster_centers_,
            SCALE_PER_CHANNEL: scales_per_channel,
            }