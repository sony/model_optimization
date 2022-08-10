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
import numpy as np
import torch


class ReshapeModel(torch.nn.Module):
    def __init__(self):
        super(ReshapeModel, self).__init__()

    def forward(self, x):
        return torch.reshape(x, [1, -1])


class SplitModel(torch.nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()

    def forward(self, x):
        x1, x2, x3 = torch.split(x, split_size_or_sections=1, dim=1)
        return x1, x2, x3


class ConcatModel(torch.nn.Module):
    def __init__(self):
        super(ConcatModel, self).__init__()

    def forward(self, x):
        return torch.concat([x, x])


class CatModel(torch.nn.Module):
    def __init__(self):
        super(CatModel, self).__init__()

    def forward(self, x):
        return torch.cat([x, x])


class DropoutModel(torch.nn.Module):
    def __init__(self):
        super(DropoutModel, self).__init__()

    def forward(self, x):
        return torch.dropout(x, 0.5, False)


class UnsqueezeModel(torch.nn.Module):
    def __init__(self):
        super(UnsqueezeModel, self).__init__()

    def forward(self, x):
        return torch.unsqueeze(x, 0)


class MeanModel(torch.nn.Module):
    def __init__(self):
        super(MeanModel, self).__init__()

    def forward(self, x):
        return torch.mean(x, 1)


class PReluModel(torch.nn.Module):
    def __init__(self):
        super(PReluModel, self).__init__()

    def forward(self, x):
        return torch.prelu(x, torch.randn(1, device='cuda'))