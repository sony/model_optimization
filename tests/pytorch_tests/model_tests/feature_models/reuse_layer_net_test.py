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
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This test checks:
The reuse of a layer in a model.
"""

class ReuseLayerNet(torch.nn.Module):
    def __init__(self):
        super(ReuseLayerNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.identity = torch.nn.Identity()

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.identity(x)
        x = self.conv1(x)
        x = self.identity(x)
        x = self.conv1(x)
        x = self.identity(x)
        x = self.conv2(x)
        x = self.identity(x)
        x = self.conv2(x)
        x = self.identity(x)
        x = self.conv2(x)
        x = self.identity(x)
        y = self.conv2(y)
        y = self.identity(y)
        y = self.conv2(y)
        y = self.identity(y)
        y = self.conv1(y)
        y = self.identity(y)
        y = self.conv1(y)
        y = self.identity(y)
        y = self.conv1(y)
        return x - y, y - x


class ReuseLayerNetTest(BasePytorchTest):
    """
    This test checks:
    The reuse of a layer in a model.
    """
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32], [self.val_batch_size, 3, 32, 32]]

    def create_feature_network(self, input_shape):
        return ReuseLayerNet()