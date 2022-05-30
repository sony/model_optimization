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
import torch
import numpy as np

from model_compression_toolkit.core.pytorch.constants import CUDA, CPU


def get_working_device() -> str:
    """
    Get the working device of the environment

    Returns:
        Device "cuda" if GPU is available, else "cpu"

    """
    return torch.device(CUDA if torch.cuda.is_available() else CPU)


def set_model(model: torch.nn.Module):
    """
    Set model to work in eval mode and GPU mode if GPU is available

    Args:
        model: Pytorch model

    Returns:

    """
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()


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
    else:
        raise Exception(f'Conversion of type {type(tensor)} to {type(torch.Tensor)} is not supported')


def torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
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
        raise Exception(f'Conversion of type {type(tensor)} to {type(np.ndarray)} is not supported')
