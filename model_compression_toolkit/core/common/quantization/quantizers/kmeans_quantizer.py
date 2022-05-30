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

from sklearn.cluster import KMeans
import numpy as np

from model_compression_toolkit.core.common.constants import CLUSTER_CENTERS, MIN_THRESHOLD, SCALE_PER_CHANNEL
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import kmeans_assign_clusters


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
    eps = 1e-8
    cluster_centers = quantization_params[CLUSTER_CENTERS]
    scales_per_channel = quantization_params[SCALE_PER_CHANNEL]
    tensor = (tensor_data / (scales_per_channel + eps))
    shape_before_kmeans = tensor.shape
    cluster_assignments = kmeans_assign_clusters(cluster_centers, tensor.reshape(-1, 1))
    quant_tensor = cluster_centers[cluster_assignments].reshape(shape_before_kmeans)
    if per_channel:
        quant_tensor = (quant_tensor * scales_per_channel)
    return quant_tensor
