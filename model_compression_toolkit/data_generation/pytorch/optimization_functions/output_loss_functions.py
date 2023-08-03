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

from model_compression_toolkit.data_generation.common.enums import OutputLossType
from model_compression_toolkit.data_generation.pytorch.constants import DEVICE


def min_max_diff(
        output_imgs: Tensor,
        output_loss_multiplier: float) -> Tensor:
    """
    Calculate the minimum-maximum difference of output images.

    Args:
        output_imgs (Tensor or List[Tensor]): The output of the model on images.
        output_loss_multiplier (float): The output loss multiplier.

    Returns:
        Tensor: The computed minimum-maximum difference loss.
    """
    if output_loss_multiplier == 0:
        return torch.zeros(1).to(DEVICE)

    if not isinstance(output_imgs, (list, tuple)):
        output_imgs = [output_imgs]
    output_loss = 0
    for output in output_imgs:
        output = torch.reshape(output, [output.shape[0], -1])
        output_loss += 1 / torch.mean(torch.max(output, 1)[0] - torch.min(output, 1)[0])
    return output_loss


def no_output_loss(
        output_imgs: Tensor,
        output_loss_multiplier: float) -> Tensor:
    """
    Calculate no output loss.

    Args:
        output_imgs (Tensor): The output of the model on images.
        output_loss_multiplier (float): The output loss multiplier.

    Returns:
        Tensor: A tensor with zero value for the loss.
    """
    return torch.zeros(1).to(DEVICE)


# Dictionary of output loss functions
output_loss_function_dict: Dict[OutputLossType, Callable] = {
    OutputLossType.NONE: no_output_loss,
    OutputLossType.MIN_MAX_DIFF: min_max_diff,
}