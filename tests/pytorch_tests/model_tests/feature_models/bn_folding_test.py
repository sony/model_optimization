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
This test checks the BatchNorm folding feature, plus adding a residual connection.
"""
class BNFoldingNet(torch.nn.Module):
    def __init__(self):
        super(BNFoldingNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = torch.relu(x)
        return x + inp


class BNFoldingNetTest(BasePytorchTest):
    """
    This test checks the BatchNorm folding feature, plus adding a residual connection.
    """
    def __init__(self, unit_test, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)

    def create_feature_network(self, input_shape):
        return BNFoldingNet()