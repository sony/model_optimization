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
import torch
from torch import Tensor
import numpy as np
from typing import Union, Optional, List, Tuple, Any

from model_compression_toolkit.core.pytorch.constants import MAX_FLOAT16, MIN_FLOAT16
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.logger import Logger


def set_model(model: torch.nn.Module, train_mode: bool = False):
    """
    Set model to work in train/eval mode and GPU mode if GPU is available

    Args:
        model: Pytorch model
        train_mode: Whether train mode or eval mode
    Returns:

    """
    if train_mode:
        model.train()
    else:
        model.eval()

    device = get_working_device()
    model.to(device)


def to_torch_tensor(data,
                    dtype: Optional = torch.float32) -> Union[Tensor, List[Tensor], Tuple[Tensor]]:
    # TODO it would make more sense to keep the original type by default but it will break lots of existing calls
    # that count on implicit convertion
    """
    Convert data to Torch tensors and move to the working device.
    Data can be numpy or torch tensor, a scalar, or a list or a tuple of such data. In the latter case only the inner
    data is converted.

    Args:
        data: Input data
        dtype: The desired data type for the tensor. Pass None to keep the type of the input data.

    Returns:
        Torch tensor
    """

    working_device = get_working_device()

    if isinstance(data, list):
        return [to_torch_tensor(t, dtype) for t in data]

    if isinstance(data, tuple):
        return tuple(to_torch_tensor(t, dtype) for t in data)

    kwargs = {} if dtype is None else {'dtype': dtype}
    return torch.as_tensor(data, device=working_device, **kwargs)


def torch_tensor_to_numpy(tensor: Union[torch.Tensor, list, tuple]) -> Union[np.ndarray, list, tuple]:
    """
    Convert a Pytorch tensor to a Numpy array.
    Args:
        tensor: Pytorch tensor.

    Returns:
        Numpy array converted from the input tensor.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return [torch_tensor_to_numpy(t) for t in tensor]
    elif isinstance(tensor, tuple):
        return tuple([torch_tensor_to_numpy(t) for t in tensor])
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().contiguous().numpy()
    else:
        Logger.critical(f'Unsupported type for conversion to Numpy array: {type(tensor)}.')


def clip_inf_values_float16(tensor: Tensor) -> Tensor:
    """
    Clips +inf and -inf values in a float16 tensor to the maximum and minimum representable values.

    Parameters:
    tensor (Tensor): Input PyTorch tensor of dtype float16.

    Returns:
    Tensor: A tensor with +inf values replaced by the maximum float16 value,
            and -inf values replaced by the minimum float16 value.
    """
    # Check if the tensor is of dtype float16
    if tensor.dtype != torch.float16:
        return tensor

    # Create a mask for inf values (both positive and negative)
    inf_mask = torch.isinf(tensor)

    # Replace inf values with max float16 value
    tensor[inf_mask] = MAX_FLOAT16 * torch.sign(tensor[inf_mask])

    return tensor
