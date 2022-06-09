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
from torch import _assert
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This tests checks that the assert operation is being removed from the graph during quantization.
"""


class AddAssertNet(torch.nn.Module):
    def __init__(self):
        super(AddAssertNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)

    def forward(self, x, y):
        _assert(x.shape == y.shape, "Inputs should have the same shape")
        x1 = self.conv1(x)
        assert(x.ndim == 4, f'expected 4D input (got {x.ndim}D input)')
        x2 = x1 + 3
        _assert(x2.shape == x1.shape, "Tensors should have the same shape")
        y = self.conv2(y)
        return x2 - y, y - x2, x2 + y


class AddAssertNetTest(BasePytorchTest):
    """
    This tests checks that the assert operation is being removed from the graph during quantization.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32], [self.val_batch_size, 3, 32, 32]]

    def create_feature_network(self, input_shape):
        return AddAssertNet()
