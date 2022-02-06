# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
import torch

from model_compression_toolkit.common.constants import THRESHOLD, SIGNED, RANGE_MIN, RANGE_MAX
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import \
    calculate_min_max_values, calculate_delta, fix_range_to_include_zero
from model_compression_toolkit.common.quantization.quantizers.uniform_quantizers import threshold_is_power_of_two


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

    min_value, max_value = calculate_min_max_values(activation_threshold,
                                                    activation_n_bits,
                                                    activation_is_signed)

    # TODO: doesn't the scale need to have threshold (or 2*threshold) in the numerator? depend on signed/unsigned?
    #  consider using calculate_delta function from quantizers_helper
    scale = 1 / 2 ** (activation_n_bits - 1)
    return lambda x: q(x, min_value, max_value, scale, activation_n_bits)


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

    min_value, max_value = calculate_min_max_values(activation_threshold,
                                                    activation_n_bits,
                                                    activation_is_signed)

    scale = calculate_delta(activation_threshold,
                            activation_n_bits,
                            activation_is_signed)
    # scale = 1 / 2 ** (activation_n_bits - 1)
    return lambda x: q(x, min_value, max_value, scale, activation_n_bits)


def uniform_quantization(activation_n_bits: int,
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
    min_value, max_value = quantization_params.get(RANGE_MIN), quantization_params.get(RANGE_MAX)

    if min_value is None or max_value is None:
        return None

    scale = (max_value - min_value) / (2 ** activation_n_bits - 1)
    return lambda x: q(x, min_value, max_value, scale, activation_n_bits)


def q(x: torch.Tensor, min_value, max_value, scale, activation_n_bits) -> torch.Tensor:
    """
    Fake-quantize the input tensor x, using a pytorch fake-quantization node.

    Args:
        x: Input tensor to quantize.
        min_value: quantization range lower bound.
        max_value: quantization range upper bound.
        scale: quantization range scale.
        activation_n_bits: number of bits to use for quantization.

    Returns:
        The fake-quantized input tensor.
    """

    # fixing range to include zero since pytorch's fake quant doesn't take care of it
    min_value, max_value = fix_range_to_include_zero(min_value,
                                                     max_value,
                                                     activation_n_bits,
                                                     per_channel=False,
                                                     channel_axis=1  # dummy
                                                     )

    return torch.fake_quantize_per_tensor_affine(x,
                                                 scale=scale,
                                                 zero_point=0,
                                                 quant_min=int(min_value / scale),
                                                 quant_max=int(max_value / scale))
