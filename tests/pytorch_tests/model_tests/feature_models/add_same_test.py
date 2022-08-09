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
This test checks the special case of addition operation with the same input.
"""
class AddSameNet(torch.nn.Module):
    def __init__(self):
        super(AddSameNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1)
        self.bn = torch.nn.BatchNorm2d(4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return torch.add(x, x)
        # return x1 + x1, torch.add(x1, x1), x1 - x1, torch.sub(x, x1)


class AddSameNetTest(BasePytorchTest):
    """
    This test checks the special case of addition operation with the same input.
    """
    def __init__(self, unit_test, float_reconstruction_error):
        super().__init__(unit_test, float_reconstruction_error)

    def create_feature_network(self, input_shape):
        return AddSameNet()