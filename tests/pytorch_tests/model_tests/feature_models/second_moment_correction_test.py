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

from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.pytorch.constants import EPSILON_VAL
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, set_model
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from tests.pytorch_tests.tpc_pytorch import get_pytorch_test_tpc_dict

"""
This test checks the Second Moment Correction feature.
"""


def bn_weight_change(bn: torch.nn.Module):
    bw_shape = bn.weight.shape
    delattr(bn, 'weight')
    delattr(bn, 'bias')
    delattr(bn, 'running_var')
    delattr(bn, 'running_mean')
    bn.register_buffer('weight', torch.ones(bw_shape))
    bn.register_buffer('bias', torch.zeros(bw_shape))
    bn.register_buffer('running_var', torch.ones(bw_shape))
    bn.register_buffer('running_mean', torch.zeros(bw_shape))
    return bn


def conv_weight_change(conv: torch.nn.Module):
    conv_w_shape = conv.weight.shape
    conv_b_shape = conv.bias.shape
    delattr(conv, 'weight')
    delattr(conv, 'bias')
    conv.register_buffer('weight', torch.ones(conv_w_shape))
    conv.register_buffer('bias', torch.zeros(conv_b_shape))
    return conv


class BaseSecondMomentTest(BasePytorchTest):
    """
    This is the base test for the Second Moment Correction feature.
    """

    def __init__(self, unit_test, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)
        self.i = 0

    def generate_inputs(self, input_shapes):
        # We want to keep the same input in order to stabilize the input's statistics
        if self.i == 0:
            self.inp = to_torch_tensor([torch.normal(mean=8.0, std=0.5, size=in_shape) for in_shape in input_shapes])
            self.i += 1
        return self.inp

    def create_inputs_shape(self):
        return [[self.val_batch_size, 1, 32, 32]]

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_quantization_method': QuantizationMethod.SYMMETRIC})
        return get_pytorch_test_tpc_dict(tp_model=tp,
                                         test_name='8bit_second_moment_correction',
                                         ftp_name='second_moment_correction_pytorch_test')

    def get_quantization_configs(self):
        quant_config = self.get_quantization_config()
        quant_config.weights_second_moment_correction = True
        quant_config.weights_second_moment_iters = 200
        return {"8bit_second_moment_correction": quant_config}

    # Check the Re-fusing of the reconstructed BN
    # new_kernel = kernel * gamma/sqrt(moving_var+eps)
    # new_bias = beta + (bias - moving_mean) * *gamma/sqrt(moving_var+eps)
    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)
            quantized_model_conv1_weight = quantized_model.conv1_bn_refused.weight.detach().cpu()
            quantized_model_conv1_bias = quantized_model.conv1_bn_refused.bias.detach().cpu()
            float_model_weight = float_model.conv1.weight.detach().cpu()
            float_model_bias = float_model.conv1.bias.detach().cpu()
            float_model_gamma = float_model.bn.weight.detach().cpu()
            float_model_beta = float_model.bn.bias.detach().cpu()
            input_var = torch.var(self.inp[0]).cpu()
            input_mean = torch.mean(self.inp[0]).cpu()
            eps = EPSILON_VAL
            weight_scale = torch.sqrt(float_model_gamma + eps) / torch.sqrt(input_var + eps)

            # new_kernel = kernel * gamma/sqrt(moving_var+eps)
            # new_bias = beta + (bias - moving_mean) * *gamma/sqrt(moving_var+eps)
            calculated_kernel = float_model_weight * weight_scale
            calculated_bias = float_model_beta + (float_model_bias - input_mean) * weight_scale

            self.unit_test.assertTrue(torch.isclose(quantized_model_conv1_weight, calculated_kernel, atol=1e-1))
            self.unit_test.assertTrue(torch.isclose(quantized_model_conv1_bias, calculated_bias,
                                                    atol=1e-1))


class ConvSecondMomentNet(torch.nn.Module):
    """
    This is the test for the Second Moment Correction feature with Conv2d.
    """
    def __init__(self):
        super(ConvSecondMomentNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.conv1 = conv_weight_change(self.conv1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.bn = bn_weight_change(self.bn)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = torch.relu(x)
        return x + inp


class ConvSecondMomentNetTest(BaseSecondMomentTest):
    """
    This is the test for the Second Moment Correction feature with Conv2d.
    """
    def create_feature_network(self, input_shape):
        return ConvSecondMomentNet()


class ConvTSecondMomentNet(torch.nn.Module):
    """
    This is the test for the Second Moment Correction feature with ConvTranspose2d.
    """
    def __init__(self):
        super(ConvTSecondMomentNet, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1)
        self.conv1 = conv_weight_change(self.conv1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.bn = bn_weight_change(self.bn)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = torch.relu(x)
        return x + inp


class ConvTSecondMomentNetTest(BaseSecondMomentTest):
    """
    This is the test for the Second Moment Correction feature with ConvTranspose2d.
    """
    def create_feature_network(self, input_shape):
        return ConvTSecondMomentNet()


class MultipleInputsConvSecondMomentNet(torch.nn.Module):
    """
    This is the test for the Second Moment Correction feature with Multiple Inputs.
    """
    def __init__(self):
        super(MultipleInputsConvSecondMomentNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.conv1 = conv_weight_change(self.conv1)
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.conv2 = conv_weight_change(self.conv2)
        self.conv3 = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.conv3 = conv_weight_change(self.conv3)
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.bn1 = bn_weight_change(self.bn1)
        self.bn2 = torch.nn.BatchNorm2d(1)
        self.bn2 = bn_weight_change(self.bn2)
        self.bn3 = torch.nn.BatchNorm2d(1)
        self.bn4 = bn_weight_change(self.bn3)

    def forward(self, x, y, z):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = torch.relu(x1)
        y1 = self.conv2(y)
        y1 = self.bn2(y1)
        y1 = torch.relu(y1)
        z1 = self.conv3(z)
        z1 = self.bn3(z1)
        z1 = torch.relu(z1)
        return x1 + y1 + z1


class MultipleInputsConvSecondMomentNetTest(BaseSecondMomentTest):
    """
    This is the test for the Second Moment Correction feature with Multiple Inputs.
    """
    def create_feature_network(self, input_shape):
        return MultipleInputsConvSecondMomentNet()

    def create_inputs_shape(self):
        return [[self.val_batch_size, 1, 32, 32],
                [self.val_batch_size, 1, 32, 32],
                [self.val_batch_size, 1, 32, 32]]

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        eps = EPSILON_VAL
        set_model(float_model)
        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)
            quantized_model_conv1_weight = quantized_model.conv1_bn_refused.weight.detach().cpu()
            quantized_model_conv1_bias = quantized_model.conv1_bn_refused.bias.detach().cpu()
            float_model_weight1 = float_model.conv1.weight.detach().cpu()
            float_model_bias1 = float_model.conv1.bias.detach().cpu()
            float_model_gamma1 = float_model.bn1.weight.detach().cpu()
            float_model_beta1 = float_model.bn1.bias.detach().cpu()
            input_var1 = torch.var(self.inp[0]).cpu()
            input_mean1 = torch.mean(self.inp[0]).cpu()
            weight_scale1 = torch.sqrt(float_model_gamma1 + eps) / torch.sqrt(input_var1 + eps)

            # new_kernel = kernel * gamma/sqrt(moving_var+eps)
            # new_bias = beta + (bias - moving_mean) * *gamma/sqrt(moving_var+eps)
            calculated_kernel1 = float_model_weight1 * weight_scale1
            calculated_bias1 = float_model_beta1 + (float_model_bias1 - input_mean1) * weight_scale1

            quantized_model_conv2_weight = quantized_model.conv2_bn_refused.weight.detach().cpu()
            quantized_model_conv2_bias = quantized_model.conv2_bn_refused.bias.detach().cpu()
            float_model_weight2 = float_model.conv2.weight.detach().cpu()
            float_model_bias2 = float_model.conv2.bias.detach().cpu()
            float_model_gamma2 = float_model.bn2.weight.detach().cpu()
            float_model_beta2 = float_model.bn2.bias.detach().cpu()
            input_var2 = torch.var(self.inp[1]).cpu()
            input_mean2 = torch.mean(self.inp[1]).cpu()
            weight_scale2 = torch.sqrt(float_model_gamma2 + eps) / torch.sqrt(input_var2 + eps)

            # new_kernel = kernel * gamma/sqrt(moving_var+eps)
            # new_bias = beta + (bias - moving_mean) * *gamma/sqrt(moving_var+eps)
            calculated_kernel2 = float_model_weight2 * weight_scale2
            calculated_bias2 = float_model_beta2 + (float_model_bias2 - input_mean2) * weight_scale2

            quantized_model_conv3_weight = quantized_model.conv3_bn_refused.weight.detach().cpu()
            quantized_model_conv3_bias = quantized_model.conv3_bn_refused.bias.detach().cpu()
            float_model_weight3 = float_model.conv3.weight.detach().cpu()
            float_model_bias3 = float_model.conv3.bias.detach().cpu()
            float_model_gamma3 = float_model.bn3.weight.detach().cpu()
            float_model_beta3 = float_model.bn3.bias.detach().cpu()
            input_var3 = torch.var(self.inp[2]).cpu()
            input_mean3 = torch.mean(self.inp[2]).cpu()
            weight_scale3 = torch.sqrt(float_model_gamma3 + eps) / torch.sqrt(input_var3 + eps)

            # new_kernel = kernel * gamma/sqrt(moving_var+eps)
            # new_bias = beta + (bias - moving_mean) * *gamma/sqrt(moving_var+eps)
            calculated_kernel3 = float_model_weight3 * weight_scale3
            calculated_bias3 = float_model_beta3 + (float_model_bias3 - input_mean3) * weight_scale3

            self.unit_test.assertTrue(torch.isclose(quantized_model_conv1_weight, calculated_kernel1, atol=1e-1))
            self.unit_test.assertTrue(torch.isclose(quantized_model_conv1_bias, calculated_bias1, atol=1e-1))
            self.unit_test.assertTrue(torch.isclose(quantized_model_conv2_weight, calculated_kernel2, atol=1e-1))
            self.unit_test.assertTrue(torch.isclose(quantized_model_conv2_bias, calculated_bias2, atol=1e-1))
            self.unit_test.assertTrue(torch.isclose(quantized_model_conv3_weight, calculated_kernel3, atol=1e-1))
            self.unit_test.assertTrue(torch.isclose(quantized_model_conv3_bias, calculated_bias3, atol=1e-1))
