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

import tensorflow as tf
from model_compression_toolkit.core.common.constants import MIN_THRESHOLD
from typing import Tuple


def ste_ceil(x: tf.Tensor) -> tf.Tensor:
    """
    Return the ceil values of a tensor.
    """
    error = tf.stop_gradient(tf.math.ceil(x) - x)
    return error + x


def ste_round(x: tf.Tensor) -> tf.Tensor:
    """
    Return the rounded values of a tensor.
    """
    error = tf.stop_gradient(tf.math.round(x) - x)
    return error + x


def log2(x: tf.Tensor) -> tf.Tensor:
    """
    Compute log2 of a tensor.
    """
    return tf.experimental.numpy.log2(x)


def power_of_two_max(max_tensor: tf.Tensor) -> tf.Tensor:
    """
    Compute the power of two threshold for a tensor.
    """
    _2 = tf.convert_to_tensor(2.0, tf.float64)
    return tf.cast(tf.math.pow(_2, ste_ceil(log2(tf.maximum(tf.cast(max_tensor, tf.float64),
                                                            MIN_THRESHOLD)))), tf.float32)


def calculate_delta(max_tensor: tf.Tensor,
                    num_bits: int,
                    signed: bool) -> tf.Tensor:
    """
    Compute the step size for the quantization.
    """
    return max_tensor / (2 ** (num_bits - int(signed)))


def adjustable_steps(x: tf.Variable, t: float) -> tf.Tensor:
    """
    A function to gradually quantize a float variable to an integer of values [-1, 0 ,1]
    Args:
        x: input float variable
        t: temperature to control quantization

    Returns:
        semi-quantized variable

    """
    return tf.sigmoid(tf.add(x, 1) / t) + tf.sigmoid(tf.add(x, -1) / t) - 1


def ste_clip(x: [tf.Tensor, tf.Variable], max_val=1, min_val=None) -> tf.Tensor:
    """
    clip a variable between fixed values such that min_val<=output<=max_val
    Args:
        x: input variable
        max_val: maximum value for clipping
        min_val: minimum value for clipping (defaults to -max_val)

    Returns:
        clipped variable

    """
    min_val = -max_val if min_val is None else min_val
    return tf.stop_gradient(tf.math.minimum(tf.math.maximum(x, min_val), max_val) - x) + x


def clip(x: [tf.Tensor, tf.Variable], max_val=1, min_val=None) -> tf.Tensor:
    """
    clip a variable between fixed values such that min_val<=output<=max_val
    Args:
        x: input variable
        max_val: maximum value for clipping
        min_val: minimum value for clipping (defaults to -max_val)
    Returns:
        clipped variable
    """
    min_val = -max_val if min_val is None else min_val
    return tf.math.minimum(tf.math.maximum(x, min_val), max_val)


def fix_range_to_include_zero(range_min: tf.Tensor,
                              range_max: tf.Tensor,
                              n_bits: int) -> Tuple[tf.Tensor, tf.Tensor]:
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
    mid_range = tf.logical_and(tf.logical_not(min_positive), tf.logical_not(max_negative))
    min_positive = tf.cast(min_positive, tf.float32)
    max_negative = tf.cast(max_negative, tf.float32)
    mid_range = tf.cast(mid_range, tf.float32)

    scale = (range_max - range_min) / (2 ** n_bits - 1)
    min_range_adj = scale * tf.round(range_min / scale)
    max_range_adj = range_max - range_min + min_range_adj

    min_range_adj = min_range_adj * mid_range + max_negative * range_min
    max_range_adj = max_range_adj * mid_range + min_positive * range_max
    return min_range_adj, max_range_adj
