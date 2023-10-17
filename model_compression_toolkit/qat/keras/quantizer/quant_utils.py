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

import tensorflow as tf
from typing import Tuple


def ste_round(x: tf.Tensor) -> tf.Tensor:
    """
    Return the rounded values of a tensor.
    """
    error = tf.stop_gradient(tf.math.round(x) - x)
    return error + x


def grad_scale(x: tf.Tensor, scale=1.0) -> tf.Tensor:
    """
    Return x in forward and x*scale in backward (for scaling the gradients).
    """
    x_scaled = scale * x
    error = tf.stop_gradient(x - x_scaled)
    return error + x_scaled


def adjust_range_to_include_zero(range_min: tf.Tensor,
                                 range_max: tf.Tensor,
                                 n_bits: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Adjusting the quantization range to include representation of 0.0 in the quantization grid.
    For per_channel quantization range_min\range_max should be tensors in the specific shape that allows
    quantization along the channel_axis.

    Args:
        range_min: min bound of the quantization range (before adjustment).
        range_max: max bound of the quantization range (before adjustment).
        n_bits: Number of bits to quantize the tensor.

    Returns: adjusted quantization range
    """
    scale = (range_max - range_min) / (2 ** n_bits - 1)
    min_range_adj = scale * tf.round(range_min / scale)
    max_range_adj = range_max - range_min + min_range_adj

    min_positive = range_min > 0
    max_negative = range_max < 0
    mid_range = tf.logical_and(tf.logical_not(min_positive), tf.logical_not(max_negative))
    min_positive = tf.cast(min_positive, tf.float32)
    max_negative = tf.cast(max_negative, tf.float32)
    mid_range = tf.cast(mid_range, tf.float32)
    min_range_adj = min_range_adj * mid_range + max_negative * range_min
    max_range_adj = max_range_adj * mid_range + min_positive * range_max

    return min_range_adj, max_range_adj
