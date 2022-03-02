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

import tensorflow as tf
from model_compression_toolkit.common.constants import MIN_THRESHOLD, THRESHOLD


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
    return tf.math.log(x) / tf.math.log(2.0)


def power_of_two_max(max_tensor: tf.Tensor) -> tf.Tensor:
    """
    Compute the power of two threshold for a tensor.
    """
    return tf.math.pow(2.0, ste_ceil(log2(tf.maximum(max_tensor, MIN_THRESHOLD))))


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


def symmetric_quantizer(input_tensor: tf.Tensor,
                        max_tensor: tf.Tensor,
                        num_bits: int,
                        signed: bool,
                        power_of_two: bool) -> tf.Tensor:
    """
    Quantize a tensor symmetrically.
    Args:
        input_tensor: Tensor to quantize.
        max_tensor: Tensor with max values to compute the threshold.
        num_bits: Num of bits to use.
        signed: Signedness of the quantization range.
        power_of_two: Whether the threshold should be constrained or not.

    Returns:
        A quantized tensor.
    """

    if power_of_two:
        max_tensor = power_of_two_max(max_tensor)
    delta = calculate_delta(max_tensor, num_bits, signed)
    tensor_q = ste_round(input_tensor / delta)
    min_int = -int(signed) * (2 ** (num_bits - int(signed)))
    max_int = (2 ** (num_bits - int(signed))) - 1
    return delta * tf.math.minimum(tf.math.maximum(tensor_q, min_int), max_int)


def symmetric_constrained_quantizer(input_tensor: tf.Tensor,
                                    auxvar_tensor: tf.Variable,
                                    max_tensor: tf.Tensor,
                                    num_bits: int,
                                    signed: bool,
                                    power_of_two: bool,
                                    max_lsbs_change: int = 1) -> tf.Tensor:
    """
    Quantize a tensor symmetrically with maximum LSBs shift.
    Args:
        input_tensor: Tensor to quantize. values of this tensor are not changed during gptq.
        auxvar_tensor: Tensor that manifests the bit shift the weight due to gptq

        max_tensor: Tensor with max values to compute the threshold.
        num_bits: Num of bits to use.
        signed: Signedness of the quantization range.
        power_of_two: Whether the threshold should be constrained or not.
        max_lsbs_change: maximum number of LSBs that the auxvar is allowed to change

    Returns:
        A quantized tensor.
    """

    if power_of_two:
        max_tensor = power_of_two_max(max_tensor)
    delta = calculate_delta(max_tensor, num_bits, signed)
    input_tensor_int = tf.stop_gradient(tf.round(input_tensor / delta))
    tensor_q = ste_round(input_tensor_int + ste_clip(auxvar_tensor, max_val=max_lsbs_change*delta)/delta)
    min_int = -int(signed) * (2 ** (num_bits - int(signed)))
    max_int = (2 ** (num_bits - int(signed))) - 1
    return delta * ste_clip(tensor_q, max_val=max_int, min_val=min_int)
