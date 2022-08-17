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
import torch.nn.functional as F
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from model_compression_toolkit.core.pytorch.utils import get_working_device

"""
This tests check the batch_norm function and demonstrates the usage of BufferHolder node.
"""


class BNFNet(torch.nn.Module):
    def __init__(self):
        super(BNFNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)
        self.running_mean = torch.zeros(4, device=get_working_device(), requires_grad=False)
        self.running_var = torch.ones(4, device=get_working_device(), requires_grad=False)

    def forward(self, x, y):
        x = self.conv1(x)
        x = F.batch_norm(x, running_mean=self.running_mean, running_var=self.running_var)
        y = self.conv2(y)
        return x - y, y - x, x + y


class BNFNetTest(BasePytorchTest):
    """
    This tests check the batch_norm function and demonstrates the usage of BufferHolder node.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32], [self.val_batch_size, 3, 32, 32]]

    def create_feature_network(self, input_shape):
        return BNFNet()
