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
from typing import Tuple
import torch


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate the rounded values of a tensor
    Args:
        x: input variable
    Returns:
        rounded value
    """
    return (torch.round(x) - x).detach() + x


def ste_clip(x: torch.Tensor, min_val=-1.0, max_val=1.0) -> torch.Tensor:
    """
    Clip a variable between fixed values such that min_val<=output<=max_val
    Args:
        x: input variable
        min_val: minimum value for clipping
        max_val: maximum value for clipping
    Returns:
        clipped variable
    """
    return (torch.clip(x, min=min_val, max=max_val) - x).detach() + x


def fix_range_to_include_zero(range_min: torch.Tensor,
                              range_max: torch.Tensor,
                              n_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
    mid_range = torch.logical_and(torch.logical_not(min_positive), torch.logical_not(max_negative))
    min_positive = min_positive.float()
    max_negative = max_negative.float()
    mid_range = mid_range.float()

    scale = (range_max - range_min) / (2 ** n_bits - 1)
    min_range_adj = scale * torch.round(range_min / scale)
    max_range_adj = range_max - range_min + min_range_adj

    min_range_adj = min_range_adj * mid_range + max_negative * range_min
    max_range_adj = max_range_adj * mid_range + min_positive * range_max
    return min_range_adj, max_range_adj


def symmetric_quantizer(tensor_data: torch.Tensor,
                        threshold: torch.Tensor,
                        n_bits: int,
                        sign: bool = False) -> torch.Tensor:
    """
    Quantize a tensor according to the number of bits and threshold.
    Symmetric quantization.
    Args:
        tensor_data: Tensor values to quantize.
        threshold: threshold for quantization.
        n_bits: Number of bits to quantize the tensor.
        sign: sign of tensor_data
    Returns:
        Quantized data.
    """

    # Compute the step size of quantized values.
    delta_tensor = threshold / (2 ** n_bits - int(sign))

    # Compute min/max int value
    min_val = -int(sign) * (2 ** (n_bits - int(sign)))
    max_val = (2 ** (n_bits - int(sign))) - 1

    # Apply rounding
    input_tensor_int = ste_round(tensor_data / delta_tensor)

    # Clip data in range
    clipped_tensor = ste_clip(input_tensor_int, min_val=min_val, max_val=max_val)

    # Quantize the data between -threshold/threshold
    q = delta_tensor * clipped_tensor
    return q


def uniform_quantizer(tensor_data: torch.Tensor,
                      range_min: torch.Tensor,
                      range_max: torch.Tensor,
                      n_bits: int) -> torch.Tensor:
    """
    Quantize a tensor according to given range (min, max) and number of bits.
    Uniform quantization.
    Args:
        tensor_data: Tensor values to quantize.
        range_min: minimum bound of the range for quantization (or array of min values per channel).
        range_max: maximum bound of the range for quantization (or array of max values per channel).
        n_bits: Number of bits to quantize the tensor.
    Returns:
        Quantized data.
    """
    # adjusts the quantization rage so the quantization grid include zero.
    a, b = fix_range_to_include_zero(range_min, range_max, n_bits)

    # Compute the step size of quantized values.
    delta_tensor = (b - a) / (2 ** n_bits - 1)

    # Apply rounding
    input_tensor_int = ste_round((tensor_data - a) / delta_tensor)

    # Clip data in range
    clipped_tensor = ste_clip(input_tensor_int, min_val=0, max_val=2 ** n_bits - 1)

    # Quantize the data between min/max of quantization range.
    q = delta_tensor * clipped_tensor + a
    return q
