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
That we support taking outputs from the middle of the model.
"""
class OutputInTheMiddleNet(torch.nn.Module):
    def __init__(self):
        super(OutputInTheMiddleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.identity = torch.nn.Identity()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.identity(x1)
        x3 = self.conv2(x2)
        x4 = torch.relu(x3)
        return x, x1, x2, x3, x4


class OutputInTheMiddleNetTest(BasePytorchTest):
    """
    This test checks:
    That we support taking outputs from the middle of the model.
    """
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        return OutputInTheMiddleNet()