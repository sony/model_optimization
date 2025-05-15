# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from torch import nn

from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from tests_pytest._fw_tests_common_base.base_ru_integration_test import BaseRUIntegrationTester
from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import TorchFwMixin


class TestRUIntegrationTorch(BaseRUIntegrationTester, TorchFwMixin):
    def test_compute_ru(self):
        super().test_compute_ru()

    def test_mult_output_activation(self):
        super().test_mult_output_activation()

    def test_snc_fusing(self):
        super().test_snc_fusing()

    def _data_gen(self):
        bchw_shape = tuple([self.bhwc_input_shape[i] for i in (0, 3, 1, 2)])
        return self.get_basic_data_gen([bchw_shape])()

    def _build_sequential_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5)
                self.dw_conv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, groups=8)
                self.conv_tr = nn.ConvTranspose2d(in_channels=16, out_channels=12, kernel_size=5)
                self.flatten = nn.Flatten()
                self.linear = nn.Linear(in_features=16*16*12, out_features=10)

            def forward(self, x):
                x = self.conv(x)
                x = x + to_torch_tensor(torch.ones((8, 14, 14)))
                x = self.dw_conv(x)
                x = self.conv_tr(x)
                x = self.flatten(x)
                x = self.linear(x)
                return x

        return Model()

    def _build_mult_output_activation_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3, groups=3)
                self.conv2 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3, groups=3)
                self.flatten = nn.Flatten()
                self.linear = nn.Linear(in_features=16*16*15, out_features=10)

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                x = x1 - x2
                x = self.flatten(x)
                x = self.linear(x)
                return x

        return Model()

    def _build_snc_model(self):
        class SncModel(nn.Module):
            def __init__(self, input_channels=3):
                super(SncModel, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=3, padding=1)
                self.activation = nn.SiLU()  # Swish activation
                self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3)
                self.conv3 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)

            def forward(self, x):
                y = self.conv1(x)
                y = self.activation(y)
                x = x+y
                x = self.conv2(x)
                x = self.activation(x)
                x = self.conv3(x)
                x = self.activation(x)
                return x

        return SncModel()

