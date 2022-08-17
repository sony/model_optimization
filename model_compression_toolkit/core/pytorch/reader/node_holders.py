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

from model_compression_toolkit.core.pytorch.constants import PLACEHOLDER, CONSTANT, BUFFER


class DummyPlaceHolder(torch.nn.Module):
    """
    Class for PlaceHolder operator since a Pytorch model doesn't have one but FX does.
    """

    def __name__(self):
        return PLACEHOLDER

    def forward(self, x):
        return x


class ConstantHolder(torch.nn.Module):
    """
    Class for saving constant values or parameters in graph inference.
    """

    def __init__(self, const_size):
        super(ConstantHolder, self).__init__()
        setattr(self, CONSTANT, torch.nn.Parameter(torch.empty(const_size)))

    def __name__(self):
        return CONSTANT

    def forward(self):
        return getattr(self, CONSTANT)


class BufferHolder(torch.nn.Module):
    """
    Class for saving buffers in graph inference.
    """

    def __init__(self, name):
        super(BufferHolder, self).__init__()
        setattr(self, BUFFER, name)

    def __name__(self):
        return BUFFER

    def forward(self):
        return self.get_buffer(getattr(self, BUFFER))

