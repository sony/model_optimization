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
This test checks that we support the connecting the input tensor to several layers
and taking them as outputs
"""
class MultipleOutputsMultipleTensorsNet(torch.nn.Module):
    def __init__(self):
        super(MultipleOutputsMultipleTensorsNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.linear = torch.nn.Linear(3*32*32, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.linear(torch.flatten(x, 1))
        x3 = torch.relu(x)
        x4 = self.conv2(x)
        x4 = torch.relu(x4)
        return x, x1, x2, x3, x4


class MultipleOutputsMultipleTensorsNetTest(BasePytorchTest):
    """
    This test checks that we support the connecting the input tensor to several layers
    and taking them as outputs
    """
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        return MultipleOutputsMultipleTensorsNet()