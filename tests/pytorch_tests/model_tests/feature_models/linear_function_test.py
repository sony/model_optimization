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
import torch.nn.functional as F
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device

"""
This test checks the linear functional substitution function.
"""


class LinearFNet(torch.nn.Module):
    def __init__(self):
        super(LinearFNet, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=1000, out_features=100, bias=False)
        self.fc2 = torch.nn.Linear(in_features=100, out_features=50, bias=True)
        self.fc3 = torch.nn.Linear(in_features=50, out_features=10, bias=False)

    def forward(self, x):
        x = F.linear(x, self.fc1.weight, self.fc1.bias)
        x = F.linear(x, bias=self.fc2.bias, weight=self.fc2.weight)
        y = F.linear(x, self.fc3.weight, bias=None)
        return y


class LinearFNetTest(BasePytorchTest):
    """
    This test check the linear functional substitution function.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 1000]]

    def create_feature_network(self, input_shape):
        return LinearFNet()
