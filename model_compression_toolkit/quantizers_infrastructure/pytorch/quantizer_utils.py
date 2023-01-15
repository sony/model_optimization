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
from typing import Tuple

import torch
import numpy as np


def get_working_device():
    """
    Get the working device of the environment

    Returns:
        Device "cuda" if GPU is available, else "cpu"

    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_torch_tensor(tensor):
    """
    Convert a Numpy array to a Torch tensor.
    Args:
        tensor: Numpy array.

    Returns:
        Torch tensor converted from the input Numpy array.
    """
    working_device = get_working_device()
    if isinstance(tensor, torch.Tensor):
        return tensor.to(working_device)
    elif isinstance(tensor, list):
        return [to_torch_tensor(t) for t in tensor]
    elif isinstance(tensor, tuple):
        return (to_torch_tensor(t) for t in tensor)
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor.astype(np.float32)).to(working_device)
    elif isinstance(tensor, float):
        return torch.Tensor([tensor]).to(working_device)
    elif isinstance(tensor, int):
        return torch.Tensor([tensor]).int().to(working_device)
    else:
        raise Exception(f'Conversion of type {type(tensor)} to {type(torch.Tensor)} is not supported')



def get_activation_symmetric_quantization_range_and_scale(activation_is_signed: bool,
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


