# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This test checks the MatMul substitution function.
"""


class MatMulFNet(torch.nn.Module):
    """
    Model for testing MatMul function
    """
    def __init__(self):
        super(MatMulFNet, self).__init__()

    def forward(self, x):
        x_t = torch.transpose(x, -1, -2)
        x = torch.matmul(x, x_t)
        return x

class MatMulOpNet(MatMulFNet):
    """
    Model for testing MatMul operator
    """
    def forward(self, x):
        x_t = torch.transpose(x, -1, -2)
        x = x @ x_t
        return x


class MatMulNetBaseTest(BasePytorchTest):
    """
    Base test for testing MatMul decomposition
    """
    def __init__(self, unit_test, input_shape):
        super().__init__(unit_test)
        self.input_shape = input_shape

    def create_inputs_shape(self):
        return [self.input_shape]

class MatMulFNetTest(MatMulNetBaseTest):
    """
    This test uses the MatMul function
    """
    def __init__(self, unit_test, input_shape):
        super().__init__(unit_test, input_shape)

    def create_feature_network(self, input_shape):
        return MatMulFNet()

class MatMulOpNetTest(MatMulNetBaseTest):
    """
    This test uses the MatMul operator - @
    """
    def __init__(self, unit_test, input_shape):
        super().__init__(unit_test, input_shape)

    def create_feature_network(self, input_shape):
        return MatMulOpNet()
