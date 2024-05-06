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

from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest


class RemoveIdentityNet(torch.nn.Module):
    def __init__(self):
        super(RemoveIdentityNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.identity = torch.nn.Identity()
        self.bn1 = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.identity(x)
        x = self.bn1(x)
        x = self.identity(x)
        return x


class RemoveIdentityTest(BasePytorchFeatureNetworkTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        return RemoveIdentityNet()

    def compare(self,
                quantized_model,
                float_model,
                input_x=None,
                quantization_info=None):
        for n,m in quantized_model.named_modules():
            # make sure identity was removed and bn was folded into the conv
            self.unit_test.assertFalse(isinstance(m, torch.nn.Identity) or isinstance(m, torch.nn.BatchNorm2d))

