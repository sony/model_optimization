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
from typing import Union
import torch
from torch.nn.functional import softmax, log_softmax, one_hot
from model_compression_toolkit.core.common.constants import MIN_THRESHOLD
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor


def power_of_two_max(max_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the power of two threshold for a tensor.
    """
    return torch.pow(2, ste_ceil(torch.log2(torch.clip(max_tensor, min=MIN_THRESHOLD, max=torch.inf))))


def ste_ceil(x: torch.Tensor) -> torch.Tensor:
    """
    Return the ceil values of a tensor.
    """
    return (torch.ceil(x) - x).detach() + x


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


def gumbel_softmax(x: torch.Tensor, tau: Union[torch.Tensor,float], gumbel_tensor: Union[torch.Tensor,float], eps: float = 1e-6, axis=0) -> torch.Tensor:
    """
    A gumbel softmax function.
    Args:
        x: A tensor of log probability.
        tau: A temperature tensor.
        gumbel_tensor: A tensor of gumbel random variable.
        eps: A small number for numeric stability.
        axis: A integer representing the axis of which the gumbel softmax applyed on.

    Returns: A gumbel softmax probability tensor.

    """
    return softmax((log_softmax(x, dim=axis) + gumbel_tensor) / (tau + eps), dim=axis)


def select_gumbel(prob: torch.Tensor) -> torch.Tensor:
    """
    This function apply ste on the output of the gumbel softmax.
    Args:
        prob: A tensor of probability.

    Returns: A Tensor of ohe hot vector

    """
    max_index = torch.argmax(prob, dim=0)
    axis_list = [i for i in range(len(max_index.shape))]
    axis_list.insert(0, len(max_index.shape))
    one_hot_prob = torch.permute(one_hot(max_index, num_classes=prob.shape[0]), axis_list)
    return one_hot_prob + 0*prob


def ste_gumbel(prob: torch.Tensor) -> torch.Tensor:
    """
    This function apply ste on the output of the gumbel softmax.
    Args:
        prob:A tensor of probability

    Returns: A Tensor of ohe hot vector with STE.

    """
    delta = (select_gumbel(prob) - prob).detach()
    return prob + delta


def sample_gumbel(shape, eps=1e-6) -> torch.Tensor:
    """
    A function that sample a tensor of i.i.d gumbel random variable.
    Args:
        shape: The tensor output shape
        eps: A small number for numeric stability.

    Returns: A tensor of i.i.d gumbel random variable.

    """
    u = to_torch_tensor(torch.rand(shape))
    return -torch.log(-torch.log(u + eps) + eps)