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



from typing import Tuple, Callable

import tensorflow as tf
import numpy as np
from tensorflow.python.util.object_identity import Reference as TFReference

from model_compression_toolkit.common.constants import THRESHOLD, SIGNED, RANGE_MIN, RANGE_MAX
from model_compression_toolkit.common.quantization.quantizers.uniform_quantizers import threshold_is_power_of_two


def quantizer_min_max_calculator(threshold: np.ndarray,
                                 num_bits: int,
                                 signed: bool) -> Tuple[float, float]:
    """
    Compute quantization range's min/max values given a threshold, number of bits,
     and whether it's signed or not.

    Args:
        threshold: Threshold for quantization range values.
        num_bits: Number of bits to use for quantization.
        signed: Whether the quantization range should include negative values or not.

    Returns:
        Min and max values for quantization range.
    """

    if signed:
        delta = threshold / (2 ** (num_bits - 1))
        min_value = -threshold
    else:
        delta = threshold / (2 ** (num_bits))
        min_value = 0

    max_value = threshold - delta
    return min_value, max_value


def power_of_two_quantization(activation_n_bits: int,
                              quantization_params: dict) -> Callable:
    """
    Use a NodeQuantizationConfig to compute a quantizer min/max values, and use it to
    build and return a fake-quantization node with power-of-two quantization.

    Args:
        activation_n_bits: Number of bits to use for quantization.
        quantization_params: Dictionary of specific parameters for this quantization function.

    Returns:
        A fake quantization node.
    """
    activation_threshold = quantization_params.get(THRESHOLD)
    activation_is_signed = quantization_params.get(SIGNED)

    if activation_threshold is None or activation_is_signed is None:
        return None
    if not threshold_is_power_of_two(activation_threshold, per_channel=False):
        return None

    min_value, max_value = quantizer_min_max_calculator(activation_threshold,
                                                        activation_n_bits,
                                                        activation_is_signed)

    return lambda x: q(x, min_value, max_value, activation_n_bits)


def symmetric_quantization(activation_n_bits: int,
                           quantization_params: dict) -> Callable:
    """
    Use a NodeQuantizationConfig to compute a quantizer min/max values, and use it to
    build and return a symmetric fake-quantization node.

    Args:
        activation_n_bits: Number of bits to use for quantization.
        quantization_params: Dictionary of specific parameters for this quantization function.

    Returns:
        A fake quantization node.
    """
    activation_threshold = quantization_params.get(THRESHOLD)
    activation_is_signed = quantization_params.get(SIGNED)

    if activation_threshold is None or activation_is_signed is None:
        return None

    min_value, max_value = quantizer_min_max_calculator(activation_threshold,
                                                        activation_n_bits,
                                                        activation_is_signed)

    return lambda x: q(x, min_value, max_value, activation_n_bits)


def uniform_quantization(activation_n_bits: int,
                         quantization_params: dict) -> Callable:
    """
    Use a NodeQuantizationConfig to compute a quantizer min/max values, and use it to
    build and return a uniform fake-quantization node.

    Args:
        activation_n_bits: Number of bits to use for quantization.
        quantization_params: Dictionary of specific parameters for this quantization function.

    Returns:
        A fake quantization node.
    """
    min_value, max_value = quantization_params.get(RANGE_MIN), quantization_params.get(RANGE_MAX)

    if min_value is None or max_value is None:
        return None

    return lambda x: q(x, min_value, max_value, activation_n_bits)


def q(x: TFReference, min_value, max_value, activation_n_bits) -> TFReference:
    """
    Fake-quantize the input tensor x, using a tensorflow fake-quantization node.

    Args:
        x: Input tensor to quantize.
        min_value: quantization range lower bound.
        max_value: quantization range upper bound.
        activation_n_bits: Number of bits to use for quantization.

    Returns:
        The fake-quantized input tensor.
    """
    return tf.quantization.fake_quant_with_min_max_vars(x,
                                                        min=min_value,
                                                        max=max_value,
                                                        num_bits=activation_n_bits)
