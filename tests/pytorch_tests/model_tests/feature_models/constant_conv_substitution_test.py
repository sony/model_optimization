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
import torch.nn as nn
import torch.nn.functional as F
import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
import numpy as np

tp = mct.target_platform


class BaseConstantConvSubstitutionTest(BasePytorchFeatureNetworkTest):

    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test)

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 32,
                                     'activation_n_bits': 32,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_pytorch_tpc(name="permute_substitution_test", tp_model=tp)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING,
                                      mct.QuantizationErrorMethod.NOCLIPPING,
                                      False, False, True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        in_torch_tensor = to_torch_tensor(input_x[0])
        y = float_model(in_torch_tensor)
        y_hat = quantized_model(in_torch_tensor)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        self.unit_test.assertTrue(np.isclose(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy(), atol=1e-8).all(), msg=f'out samples are not as expected!')


class ConstantConvSubstitutionTest(BaseConstantConvSubstitutionTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3), bias=False)
            self.conv2 = nn.Conv2d(16, 8, kernel_size=(1,1), bias=True)

        def forward(self, x):
            x = F.conv2d(x, weight=self.conv1.weight, bias=self.conv1.bias)
            x = F.relu(x)
            x = F.conv2d(x, weight=self.conv2.weight, bias=self.conv2.bias)
            x = F.relu(x)
            return x

    def create_networks(self):
        return self.ConvNet()


class ConstantConvReuseSubstitutionTest(BaseConstantConvSubstitutionTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3), bias=False)
            self.conv2 = nn.Conv2d(16, 8, kernel_size=(1,1), bias=True)
            self.conv3 = nn.Conv2d(8, 8, kernel_size=(1,1), bias=True)

        def forward(self, x):
            x1 = F.conv2d(x, weight=self.conv1.weight, bias=self.conv1.bias)
            x1 = F.hardswish(x1) + x1
            x2 = F.conv2d(x1, weight=self.conv2.weight, bias=self.conv2.bias)
            x3 = F.relu(x2) - x2
            x4 = F.conv2d(x3, weight=self.conv3.weight, bias=self.conv3.bias) + x2
            x4 = F.gelu(x4)
            y = F.conv2d(x4, weight=self.conv3.weight, bias=self.conv3.bias) + x3 # reuse weights/bias
            return y

    def create_networks(self):
        return self.ConvNet()


class ConstantConvTransposeSubstitutionTest(BaseConstantConvSubstitutionTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class ConvTransposeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.convTr1 = nn.ConvTranspose2d(3, 16, kernel_size=(3,3), bias=False)
            self.convTr2 = nn.ConvTranspose2d(16, 8, kernel_size=(1,1), bias=True)

        def forward(self, x):
            x = F.conv_transpose2d(x, weight=self.convTr1.weight, bias=self.convTr1.bias)
            x = F.relu(x)
            x = F.conv_transpose2d(x, weight=self.convTr2.weight, bias=self.convTr2.bias)
            x = F.relu(x)
            return x

    def create_networks(self):
        return self.ConvTransposeNet()
