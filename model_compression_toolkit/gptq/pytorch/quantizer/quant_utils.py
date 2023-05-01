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
from typing import Union, Tuple
import torch
from model_compression_toolkit.constants import MIN_THRESHOLD


def power_of_two_max(max_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the power of two threshold for a tensor.
    """
    return torch.pow(2, ste_ceil(torch.log2(torch.clip(max_tensor, min=MIN_THRESHOLD, max=torch.inf))))


def calculate_delta(max_tensor: torch.Tensor,
                    num_bits: int,
                    signed: bool) -> torch.Tensor:
    """
    Compute the step size for the symmetric quantization.
    """
    return max_tensor / (2 ** (num_bits - int(signed)))


def calculate_delta_uniform(min_tensor: torch.Tensor,
                            max_tensor: torch.Tensor,
                            num_bits: int) -> torch.Tensor:
    """
    Compute the step size for the uniform quantization.
    """
    return (max_tensor-min_tensor) / (2 ** num_bits - 1)


def ste_ceil(x: torch.Tensor) -> torch.Tensor:
    """
    Return the ceil values of a tensor.
    """
    return (torch.ceil(x) - x).detach() + x


def ste_floor(x: torch.Tensor) -> torch.Tensor:
    """
    Return the floor values of a tensor.
    """
    return (torch.floor(x) - x).detach() + x


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
