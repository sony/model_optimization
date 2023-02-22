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

import numpy as np
import tensorflow as tf


def lut_quantizer(tensor_data: tf.Tensor,
                  cluster_centers: np.ndarray,
                  signed: bool,
                  threshold: np.ndarray,
                  multiplier_n_bits: int,
                  eps: float) -> tf.Tensor:
    """
    Quantize a tensor using a non-uniform quantization based on the pre-defined clusters.
    1. Scales tensor_data with the threshold into multiplier_n_bits quantization range.
    2. Assigns cluster centers to each value.
    3. Scales back by multiplying the result by threshold and dividing with the quantization range max value.
    The result is the quantized tensor.

    Args:
        tensor_data: Input activation tensor.
        cluster_centers: the cluster centers to assign the tensor values.
        signed: Whether the quantization is signed or not.
        threshold: threshold for quantization.
        multiplier_n_bits: Number of bits that determines the quantization range
        eps: Small value for numerical stability in division.

    Returns: Quantized tensor.
    """

    tensor = int_quantization_with_threshold(tensor_data, n_bits=multiplier_n_bits, signed=signed, threshold=threshold,
                                             eps=eps)
    tensor = tf.expand_dims(tensor, -1)

    expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
    cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
    centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

    quant_tensor = (centers / (2 ** (multiplier_n_bits - int(signed)))) * threshold

    return quant_tensor


def int_quantization_with_threshold(data: tf.Tensor,
                                    n_bits: int,
                                    signed: bool,
                                    threshold: np.ndarray,
                                    eps: float) -> tf.Tensor:
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
