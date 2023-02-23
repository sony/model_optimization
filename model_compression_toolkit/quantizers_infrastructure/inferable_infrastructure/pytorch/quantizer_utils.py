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


def lut_quantizer(tensor_data: torch.Tensor,
                  cluster_centers: torch.Tensor,
                  signed: bool,
                  threshold: torch.Tensor,
                  multiplier_n_bits: int,
                  eps: float) -> torch.Tensor:
    """
    Quantize a tensor using a non-uniform quantization based on the pre-defined clusters.
    1. Scales tensor_data with the threshold into n-bit quantization range.
    2. Assigns cluster centers to each value.
    3. Scales back by multiplying the result by threshold and dividing with the quantization range max value.
    The result is the quantized tensor.

    Args:
        tensor_data: Input activation tensor.
        cluster_centers: The cluster centers to assign the tensor values.
        signed: Whether the quantization is signed or not.
        threshold: Threshold for quantization.
        multiplier_n_bits: Number of bits that determines the quantization range
        eps: Small value for numerical stability in division.

    Returns: Quantized tensor.
    """

    tensor = int_quantization_with_threshold(tensor_data, n_bits=multiplier_n_bits, signed=signed, threshold=threshold,
                                             eps=eps)
    tensor = tensor.unsqueeze(-1)

    expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
    cluster_assignments = torch.argmin(torch.abs(tensor - expanded_cluster_centers), dim=-1)
    centers = cluster_centers.flatten()[cluster_assignments]

    quant_tensor = (centers / (2 ** (multiplier_n_bits - int(signed)))) * threshold

    return quant_tensor


def int_quantization_with_threshold(data: torch.Tensor,
                                    n_bits: int,
                                    signed: bool,
                                    threshold: torch.Tensor,
                                    eps: float) -> torch.Tensor:
    """
    Divides data by threshold and quantize it to integers in the quantization range (depends on signed value).

    Args:
        data: Tensor data.
        n_bits: Number of bits that determines the quantization range.
        signed: Whether the quantization is signed or not.
        threshold: Threshold for quantization.
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

    return torch.clip((data / (threshold + eps)) * (2 ** (n_bits - int(signed))),
                      min=clip_min, max=clip_max)
