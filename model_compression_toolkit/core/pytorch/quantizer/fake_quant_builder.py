# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Callable
import torch

from model_compression_toolkit.constants import THRESHOLD, SIGNED, RANGE_MIN, RANGE_MAX
from model_compression_toolkit.core.common.quantization.quantizers.uniform_quantizers import threshold_is_power_of_two
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import fix_range_to_include_zero


def get_symmetric_quantization_range_and_scale(activation_is_signed: bool,
                                               activation_n_bits: int,
                                               activation_threshold: float):
    """
    Calculates lower and upper bounds on the quantization range, along with quantization scale,
    for symmetric quantization (used for the symmetric and power-of-two quantizers),
    according to whether the quantization is signed or unsigned.

    Args:
        activation_is_signed: Whether the quantization is signed or not.
        activation_n_bits: Number of bits to use for quantization.
        activation_threshold: The quantization threshold.

    Returns: range lower bound, range upper bound and quantization scale.

    """
    if activation_is_signed:
        min_value = -2 ** (activation_n_bits - 1)
        max_value = 2 ** (activation_n_bits - 1) - 1
        scale = activation_threshold / 2 ** (activation_n_bits - 1)
    else:
        min_value = 0
        max_value = (2 ** activation_n_bits) - 1
        scale = activation_threshold / 2 ** activation_n_bits

    return min_value, max_value, scale


def power_of_two_quantization(activation_n_bits: int,
                              quantization_params: dict) -> Callable:
    """
    Use a NodeQuantizationConfig to compute a quantizer min/max values, and use it to
    build and return a fake-quantization node, quantized with a power-of-two threshold.
    Args:
        activation_n_bits: Number of bits to use for quantization.
        quantization_params: Dictionary of specific parameters for this quantization function.
    Returns:
        A fake quantization node.
    """
    activation_threshold = quantization_params.get(THRESHOLD)
    activation_is_signed = quantization_params.get(SIGNED)

    if activation_threshold is None or activation_is_signed is None:
        return None # pragma: no cover
    if not threshold_is_power_of_two(activation_threshold, per_channel=False):
        return None # pragma: no cover

    min_value, max_value, scale = get_symmetric_quantization_range_and_scale(activation_is_signed,
                                                                             activation_n_bits,
                                                                             activation_threshold)

    return lambda x: q(x, min_value, max_value, scale)


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
        return None # pragma: no cover

    min_value, max_value, scale = get_symmetric_quantization_range_and_scale(activation_is_signed,
                                                                             activation_n_bits,
                                                                             activation_threshold)

    return lambda x: q(x, min_value, max_value, scale)


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
    a, b = quantization_params.get(RANGE_MIN), quantization_params.get(RANGE_MAX)

    if a is None or b is None:
        return None # pragma: no cover

    # fixing quantization range to include 0
    a = 0 if a > 0 else a
    b = 0 if b < 0 else b
    a, b = fix_range_to_include_zero(a, b, activation_n_bits)

    min_value = 0
    max_value = 2 ** activation_n_bits - 1
    scale = (b - a) / ((2 ** activation_n_bits) - 1)
    zero_point = -round(a / scale)  # zp has to be positive, and a <=0, so we multiply by -1

    return lambda x: q(x, min_value, max_value, scale, zero_point)


def q(x: torch.Tensor,
      min_value: int,
      max_value: int,
      scale: float,
      zero_point: int = 0) -> torch.Tensor:
    """
    Fake-quantize the input tensor x, using a pytorch fake-quantization node.
    Args:
        x: input tensor to quantize.
        min_value: lower bound of the quantized domain.
        max_value: upper bound of the quantized domain.
        scale: quantization scale.
        zero_point: quantization zero_point
    Returns:
        The fake-quantized input tensor.
    """

    return torch.fake_quantize_per_tensor_affine(x,
                                                 scale=scale,
                                                 zero_point=zero_point,
                                                 quant_min=min_value,
                                                 quant_max=max_value)
