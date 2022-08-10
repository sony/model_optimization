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

import numpy as np
import torch
from torch.nn import Conv2d, ReLU, ZeroPad2d, BatchNorm2d, ConvTranspose2d
from torch.nn.functional import relu

from model_compression_toolkit.core.common.substitutions.scale_equalization import fixed_second_moment_after_relu, \
    fixed_mean_after_relu
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_tp_model
from model_compression_toolkit.core.pytorch.utils import set_model
from tests.pytorch_tests.tpc_pytorch import get_pytorch_test_tpc_dict
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest


def bn_weight_change(bn: torch.nn.Module):
    bw_shape = bn.weight.shape
    delattr(bn, 'weight')
    bn.register_buffer('weight', 10 * torch.rand(bw_shape))
    return bn


"""
This test checks the Channel Scale Equalization feature.
"""


class ScaleEqualizationBaseTest(BasePytorchTest):
    def __init__(self, unit_test, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    def get_tpc(self):
        return get_pytorch_test_tpc_dict(tp_model=get_tp_model(),
                                         test_name='8bit_scale_equalization',
                                         ftp_name='8bit_scale_equalization_pytorch_test')

    def get_quantization_configs(self):
        quant_config = self.get_quantization_config()
        quant_config.activation_channel_equalization = True
        return {"8bit_scale_equalization": quant_config}

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)
            quantized_model_layer1_weight = quantized_model.layer1_bn.weight.detach().cpu().numpy().squeeze()
            quantized_model_layer2_weight = quantized_model.layer2.weight.detach().cpu().numpy().squeeze()
            float_model_layer1_weight = float_model.layer1.weight.detach().cpu().numpy().squeeze()
            float_model_layer2_weight = float_model.layer2.weight.detach().cpu().numpy().squeeze()
            gamma = float_model.bn.weight.detach().cpu().numpy().squeeze()
            bn_beta = float_model.bn.bias.detach().cpu().numpy().squeeze()

            if type(quantized_model.layer1_bn) == ConvTranspose2d:
                quantized_model_layer1_weight = np.transpose(quantized_model_layer1_weight)
                float_model_layer1_weight = np.transpose(float_model_layer1_weight)

            fixed_second_moment_vector = fixed_second_moment_after_relu(bn_beta, gamma)
            fixed_mean_vector = fixed_mean_after_relu(bn_beta, gamma)
            fixed_std_vector = np.sqrt(fixed_second_moment_vector - np.power(fixed_mean_vector, 2))

            scale_factor = 1.0 / fixed_std_vector
            scale_factor = np.minimum(scale_factor, 1.0)

            quantized_model_layer1_weight_without_bn_fold = quantized_model_layer1_weight / gamma

            alpha = quantized_model_layer1_weight_without_bn_fold / float_model_layer1_weight

            beta = float_model_layer2_weight / quantized_model_layer2_weight

            if type(quantized_model.layer1_bn) == ConvTranspose2d:
                alpha = np.transpose(alpha)
            if type(quantized_model.layer2) == ConvTranspose2d:
                beta = np.transpose(beta)

            self.unit_test.assertTrue(np.allclose(alpha, beta, atol=1e-1))
            self.unit_test.assertTrue(np.alltrue(alpha <= 1.0))
            self.unit_test.assertTrue(np.allclose(alpha, scale_factor, atol=1e-1))


class ScaleEqualizationNet(torch.nn.Module):
    def __init__(self):
        super(ScaleEqualizationNet, self).__init__()
        self.layer1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.relu1 = ReLU()
        self.layer2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = bn_weight_change(self.bn)

    def forward(self, inp):
        x = self.layer1(inp)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x


class ScaleEqualizationWithZeroPadNet(torch.nn.Module):
    def __init__(self):
        super(ScaleEqualizationWithZeroPadNet, self).__init__()
        self.layer1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.relu1 = ReLU()
        self.layer2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.zero_pad = ZeroPad2d(padding=2)
        self.bn = bn_weight_change(self.bn)

    def forward(self, inp):
        x = self.layer1(inp)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.zero_pad(x)
        x = self.layer2(x)
        return x


class ScaleEqualizationReluFuncNet(torch.nn.Module):
    def __init__(self):
        super(ScaleEqualizationReluFuncNet, self).__init__()
        self.layer1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.layer2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = bn_weight_change(self.bn)

    def forward(self, inp):
        x = self.layer1(inp)
        x = self.bn(x)
        x = relu(x)
        x = self.layer2(x)
        return x


class ScaleEqualizationReluFuncWithZeroPadNet(torch.nn.Module):
    def __init__(self):
        super(ScaleEqualizationReluFuncWithZeroPadNet, self).__init__()
        self.layer1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.layer2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = bn_weight_change(self.bn)
        self.zero_pad = ZeroPad2d(padding=2)

    def forward(self, inp):
        x = self.layer1(inp)
        x = self.bn(x)
        x = relu(x)
        x = self.zero_pad(x)
        x = self.layer2(x)
        return x


class ScaleEqualizationConvTransposeWithZeroPadNet(torch.nn.Module):
    def __init__(self):
        super(ScaleEqualizationConvTransposeWithZeroPadNet, self).__init__()
        self.layer1 = ConvTranspose2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.relu1 = ReLU()
        self.layer2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.zero_pad = ZeroPad2d(padding=2)
        self.bn = bn_weight_change(self.bn)

    def forward(self, inp):
        x = self.layer1(inp)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.zero_pad(x)
        x = self.layer2(x)
        return x


class ScaleEqualizationConvTransposeReluFuncNet(torch.nn.Module):
    def __init__(self):
        super(ScaleEqualizationConvTransposeReluFuncNet, self).__init__()
        self.layer1 = ConvTranspose2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.layer2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = bn_weight_change(self.bn)

    def forward(self, inp):
        x = self.layer1(inp)
        x = self.bn(x)
        x = relu(x)
        x = self.layer2(x)
        return x


class ScaleEqualizationReluFuncConvTransposeWithZeroPadNet(torch.nn.Module):
    def __init__(self):
        super(ScaleEqualizationReluFuncConvTransposeWithZeroPadNet, self).__init__()
        self.layer1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.layer2 = ConvTranspose2d(3, 3, kernel_size=1, stride=1)
        self.bn = bn_weight_change(self.bn)
        self.zero_pad = ZeroPad2d(padding=2)

    def forward(self, inp):
        x = self.layer1(inp)
        x = self.bn(x)
        x = relu(x)
        x = self.zero_pad(x)
        x = self.layer2(x)
        return x


class ScaleEqualizationNetTest(ScaleEqualizationBaseTest):
    """
    This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a layer
    """

    def create_feature_network(self, input_shape):
        return ScaleEqualizationNet()


class ScaleEqualizationWithZeroPadNetTest(ScaleEqualizationBaseTest):
    """
    This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a layer
    and with zero padding.
    """

    def create_feature_network(self, input_shape):
        return ScaleEqualizationWithZeroPadNet()


class ScaleEqualizationReluFuncNetTest(ScaleEqualizationBaseTest):
    """
    This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a function
    """

    def create_feature_network(self, input_shape):
        return ScaleEqualizationReluFuncNet()


class ScaleEqualizationReluFuncWithZeroPadNetTest(ScaleEqualizationBaseTest):
    """
    This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a function
    and with zero padding.
    """

    def create_feature_network(self, input_shape):
        return ScaleEqualizationReluFuncWithZeroPadNet()


class ScaleEqualizationConvTransposeWithZeroPadNetTest(ScaleEqualizationBaseTest):
    """
    This test checks the Channel Scale Equalization feature in ConvTranspose2D - Relu - Conv2D with Relu as a layer
    and with zero padding.
    """

    def create_feature_network(self, input_shape):
        return ScaleEqualizationConvTransposeWithZeroPadNet()


class ScaleEqualizationConvTransposeReluFuncNetTest(ScaleEqualizationBaseTest):
    """
    This test checks the Channel Scale Equalization feature in ConvTranspose2D - Relu - Conv2D with Relu as a function.
    """

    def create_feature_network(self, input_shape):
        return ScaleEqualizationConvTransposeReluFuncNet()


class ScaleEqualizationReluFuncConvTransposeWithZeroPadNetTest(ScaleEqualizationBaseTest):
    """
    This test checks the Channel Scale Equalization feature in Conv2D - Relu - ConvTranspose2D with Relu as a function
    and with zero padding.
    """

    def create_feature_network(self, input_shape):
        return ScaleEqualizationReluFuncConvTransposeWithZeroPadNet()
