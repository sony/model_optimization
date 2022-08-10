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
This test checks that we build a correct graph and correctly reconstruct the model
given the fact that we reuse nodes and abuse the naming convention of fx (if we resuse
"conv1" it will be displayed as "conv1_1". So we intentionally add a node named "conv1_1").
"""
class ReuseNameNet(torch.nn.Module):
    def __init__(self):
        super(ReuseNameNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv1_1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(1, 3, kernel_size=1, stride=1)
        self.identity = torch.nn.Identity()

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.identity(x)
        x = self.conv1(x)
        x = self.identity(x)
        return self.conv1(x), self.conv1_1(y)


class ReuseNameNetTest(BasePytorchTest):
    """
    This test checks that we build a correct graph and correctly reconstruct the model
    given the fact that we reuse nodes and abuse the naming convention of fx (if we resuse
    "conv1" it will be displayed as "conv1_1". So we intentionally add a node named "conv1_1").
    """
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32], [self.val_batch_size, 3, 32, 32]]

    def create_feature_network(self, input_shape):
        return ReuseNameNet()