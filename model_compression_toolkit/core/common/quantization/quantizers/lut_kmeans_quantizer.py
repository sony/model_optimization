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

from model_compression_toolkit.core.common.constants import CLUSTER_CENTERS, SCALE_PER_CHANNEL, \
    MULTIPLIER_N_BITS
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import kmeans_assign_clusters, \
    get_quantized_tensor, int_quantization_with_threshold


def lut_kmeans_quantizer(tensor_data: np.ndarray,
                        n_bits: int,
                        signed: bool,
                        quantization_params: dict,
                        per_channel: bool,
                        output_channels_axis: int) -> np.ndarray:
    """
    Quantize a tensor with given cluster centers and thresholds-per-channel vector.
    1. We divide tensor_data with the scale vector per channel.
    2. We scale the result to the range [-2^(MULTIPLIER_N_BITS-1), 2^(MULTIPLIER_N_BITS-1)-1].
    3. We assign cluster centers to every value, multiply by thresholds_per_channel and divide by 2^(MULTIPLIER_N_BITS-1).
    The result is the quantized tensor.


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
    thresholds_per_channel = quantization_params[SCALE_PER_CHANNEL]
    tensor = int_quantization_with_threshold(tensor_data, thresholds_per_channel, MULTIPLIER_N_BITS)
    shape_before_kmeans = tensor.shape
    cluster_assignments = kmeans_assign_clusters(cluster_centers, tensor.reshape(-1, 1))
    quant_tensor = get_quantized_tensor(cluster_centers[cluster_assignments].reshape(shape_before_kmeans),
                                        thresholds_per_channel,
                                        MULTIPLIER_N_BITS)
    return quant_tensor
