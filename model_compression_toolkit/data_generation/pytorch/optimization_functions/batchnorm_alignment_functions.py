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
from typing import Dict, Callable

import torch
from torch import Tensor

from model_compression_toolkit.data_generation.common.enums import BatchNormAlignemntLossType


def l2_square(bn_mean: Tensor,
              input_mean: Tensor,
              bn_std: Tensor,
              input_std: Tensor) -> Tensor:
    """
    Compute the L2 Square loss for batch normalization alignment.

    Args:
        bn_mean (Tensor): The mean of the batch normalization layer from the original statistics.
        input_mean (Tensor): The mean of the batch normalization layer from the current batch statistics.
        bn_std (Tensor): The standard deviation of the batch normalization layer from the original statistics.
        input_std (Tensor): The standard deviation of the batch normalization layer from the current batch statistics.

    Returns:
        Tensor: The L2 Square loss value for batch normalization alignment.
    """
    return torch.linalg.norm(input_mean - bn_mean) ** 2 / bn_mean.size(0) + \
           torch.linalg.norm(input_std - bn_std) ** 2 / bn_std.size(0)

# Dictionary of batch normalization alignment loss functions
bn_alignment_loss_function_dict: Dict[BatchNormAlignemntLossType, Callable] = {
    BatchNormAlignemntLossType.L2_SQUARE: l2_square,
}