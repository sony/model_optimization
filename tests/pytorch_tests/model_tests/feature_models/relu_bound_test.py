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

from model_compression_toolkit import QuantizationConfig, QuantizationErrorMethod
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_tp_model
from tests.pytorch_tests.tpc_pytorch import get_pytorch_test_tpc_dict
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from torch.nn import Conv2d, ReLU, ReLU6, Hardtanh
from torch.nn.functional import relu, relu6, hardtanh
import model_compression_toolkit as mct
import numpy as np
from model_compression_toolkit.core.pytorch.utils import set_model

"""
This test checks the BatchNorm folding feature, plus adding a residual connection.
"""


class ReLUBoundToPOTNet(torch.nn.Module):
    def __init__(self):
        super(ReLUBoundToPOTNet, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.relu1 = ReLU6()
        self.conv2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv3 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv4 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.relu2 = ReLU()
        self.identity = torch.nn.Identity()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.identity(x)
        x = self.conv3(x)
        x = relu6(x)
        x = self.conv4(x)
        x = self.relu2(x)
        x = relu(x)
        return x


class HardtanhBoundToPOTNet(torch.nn.Module):
    def __init__(self):
        super(HardtanhBoundToPOTNet, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.hardtanh1 = Hardtanh(min_val=0.0, max_val=6.0)
        self.conv2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv3 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.hardtanh2 = Hardtanh(min_val=-2.0, max_val=6.0)
        self.conv4 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv5 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv6 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv7 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.hardtanh3 = Hardtanh(min_val=0.0, max_val=4.0)
        self.conv8 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv9 = Conv2d(3, 3, kernel_size=1, stride=1)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.hardtanh1(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.conv3(x)
        x = self.hardtanh2(x)
        x = self.conv4(x)
        x = relu(x)
        x = self.conv5(x)
        x = hardtanh(x, min_val=0.0, max_val=6.0)
        x = self.conv6(x)
        x = relu(x)
        x = self.conv7(x)
        x = self.hardtanh3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = relu(x)
        return x


class ReLUBoundToPOTNetTest(BasePytorchTest):
    """
    This test checks the ReLU Bound To POT feature.
    """

    def __init__(self, unit_test, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    def get_tpc(self):
        return get_pytorch_test_tpc_dict(tp_model=get_tp_model(),
                                         test_name='8bit_relu_bound',
                                         ftp_name='relu_bound_pytorch_test')

    def get_quantization_configs(self):
        quant_config = QuantizationConfig(QuantizationErrorMethod.MSE,
                                          QuantizationErrorMethod.MSE,
                                          relu_bound_to_power_of_2=True)
        return {"8bit_relu_bound": quant_config}

    def create_feature_network(self, input_shape):
        return ReLUBoundToPOTNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)
            alpha_1 = (quantized_model.conv1.weight / float_model.conv1.weight).detach().cpu().numpy().mean()
            beta_1 = (quantized_model.conv2.weight / float_model.conv2.weight).detach().cpu().numpy().mean()
            alpha_2 = (quantized_model.conv3.weight / float_model.conv3.weight).detach().cpu().numpy().mean()
            beta_2 = (quantized_model.conv4.weight / float_model.conv4.weight).detach().cpu().numpy().mean()

            self.unit_test.assertTrue(np.allclose(alpha_1 * beta_1, 1, atol=1e-1))
            self.unit_test.assertTrue(np.allclose(alpha_1 * 6 / 8, 1, atol=1e-1))
            self.unit_test.assertTrue(np.allclose(8 / 6 * beta_1, 1, atol=1e-1))
            self.unit_test.assertTrue(np.allclose(alpha_2 * beta_2, 1, atol=1e-1))
            self.unit_test.assertTrue(np.allclose(alpha_2 * 6 / 8, 1, atol=1e-1))
            self.unit_test.assertTrue(np.allclose(8 / 6 * beta_2, 1, atol=1e-1))


class HardtanhBoundToPOTNetTest(BasePytorchTest):
    """
    This test checks the Hardtanh Bound To POT feature.
    """

    def __init__(self, unit_test, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    def get_tpc(self):
        return get_pytorch_test_tpc_dict(tp_model=get_tp_model(),
                                         test_name='8bit_relu_bound',
                                         ftp_name='relu_bound_pytorch_test')

    def get_quantization_configs(self):
        quant_config = QuantizationConfig(QuantizationErrorMethod.MSE,
                                          QuantizationErrorMethod.MSE,
                                          relu_bound_to_power_of_2=True)
        return {"8bit_relu_bound": quant_config}

    def create_feature_network(self, input_shape):
        return HardtanhBoundToPOTNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)
            alpha_1 = (quantized_model.conv1.weight / float_model.conv1.weight).detach().cpu().numpy().mean()
            beta_1 = (quantized_model.conv2.weight / float_model.conv2.weight).detach().cpu().numpy().mean()
            alpha_2 = (quantized_model.conv5.weight / float_model.conv5.weight).detach().cpu().numpy().mean()
            beta_2 = (quantized_model.conv6.weight / float_model.conv6.weight).detach().cpu().numpy().mean()

            self.unit_test.assertTrue(np.allclose(alpha_1 * beta_1, 1, atol=1e-1))
            self.unit_test.assertTrue(np.allclose(alpha_1 * 6 / 8, 1, atol=1e-1))
            self.unit_test.assertTrue(np.allclose(8 / 6 * beta_1, 1, atol=1e-1))
            self.unit_test.assertTrue(np.allclose(alpha_2 * beta_2, 1, atol=1e-1))
            self.unit_test.assertTrue(np.allclose(alpha_2 * 6 / 8, 1, atol=1e-1))
            self.unit_test.assertTrue(np.allclose(8 / 6 * beta_2, 1, atol=1e-1))

            self.unit_test.assertTrue(quantized_model.hardtanh2.max_val == float_model.hardtanh2.max_val)
            self.unit_test.assertTrue(quantized_model.hardtanh2.min_val == float_model.hardtanh2.min_val)
            self.unit_test.assertTrue(quantized_model.hardtanh3.max_val == float_model.hardtanh3.max_val)
            self.unit_test.assertTrue(quantized_model.hardtanh3.min_val == float_model.hardtanh3.min_val)
