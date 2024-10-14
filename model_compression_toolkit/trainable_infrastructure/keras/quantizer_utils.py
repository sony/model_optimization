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

from model_compression_toolkit.qat.keras.quantizer.quant_utils import grad_scale, ste_round, \
    adjust_range_to_include_zero


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


def symmetric_lsq_quantizer(x: tf.Tensor,
                            thresholds: tf.Tensor,
                            num_bits: int,
                            sign: bool,
                            min_int: int,
                            max_int:int,
                            scale_factor: float) -> tf.Tensor:
    """
    Symmetric quantizer according to LSQ algorithm: https://arxiv.org/pdf/1902.08153.pdf
    Args:
        x: input to quantize
        thresholds: thresholds of quantization levels
        num_bits: number of bits for quantization
        sign: whether x is signed or not
        min_int: min clipping integer value
        max_int: max clipping integer value
        scale_factor: grad scale of LSQ algorithm
    Returns:
        A quantized tensor
    """
    delta = thresholds / (2 ** (num_bits - int(sign)))
    delta_scaled = grad_scale(delta, scale_factor)
    rounded = ste_round(x / delta_scaled)
    clipped = tf.math.minimum(tf.math.maximum(rounded, min_int), max_int)
    quantized = delta_scaled * clipped
    return quantized


def uniform_lsq_quantizer(x: tf.Tensor,
                          min_range: tf.Tensor,
                          max_range: tf.Tensor,
                          num_bits: int,
                          min_int: int,
                          max_int:int,
                          scale_factor: float) -> tf.Tensor:
    """
    Uniform quantizer according to LSQ algorithm: https://arxiv.org/pdf/1902.08153.pdf
    Args:
        x: input to quantize
        min_range: min range of quantization values
        max_range: min range of quantization values
        num_bits: number of bits for quantization
        min_int: min clipping integer value
        max_int: max clipping integer value
        scale_factor: grad scale of LSQ algorithm
    Returns:
        A quantized tensor
    """
    min_range, max_range = adjust_range_to_include_zero(min_range, max_range, num_bits)
    delta = (max_range - min_range) / (2 ** num_bits - 1)
    delta_scaled = grad_scale(delta, scale_factor)
    rounded = ste_round((x-min_range) / delta_scaled)
    clipped = tf.math.minimum(tf.math.maximum(rounded, min_int), max_int)
    quantized = delta_scaled * clipped + min_range
    return quantized
