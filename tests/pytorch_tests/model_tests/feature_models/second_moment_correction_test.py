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
import copy
import random
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.nn import Module

from model_compression_toolkit.core import FrameworkInfo, CoreConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.statistics_correction.apply_second_moment_correction_to_graph import \
    quantized_model_builder_for_second_moment_correction
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.core.pytorch.constants import EPSILON_VAL, GAMMA, BETA, MOVING_MEAN, MOVING_VARIANCE
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.core.pytorch.statistics_correction.apply_second_moment_correction import \
    pytorch_apply_second_moment_correction
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, set_model
from model_compression_toolkit.core.runner import _init_tensorboard_writer, core_runner
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
        self.num_calibration_iter = 200
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

    def get_core_configs(self):
        core_config = self.get_core_config()
        core_config.quantization_config.weights_second_moment_correction = True
        return {"8bit_second_moment_correction": core_config}

    # Check the Re-fusing of the reconstructed BN
    # new_kernel = kernel * gamma/sqrt(moving_var+eps)
    # new_bias = beta + (bias - moving_mean) * *gamma/sqrt(moving_var+eps)
    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)
            quantized_model_conv1_weight = quantized_model.conv1_bn_refused.layer.weight.detach().cpu()
            quantized_model_conv1_bias = quantized_model.conv1_bn_refused.layer.bias.detach().cpu()
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
            quantized_model_conv1_weight = quantized_model.conv1_bn_refused.layer.weight.detach().cpu()
            quantized_model_conv1_bias = quantized_model.conv1_bn_refused.layer.bias.detach().cpu()
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

            quantized_model_conv2_weight = quantized_model.conv2_bn_refused.layer.weight.detach().cpu()
            quantized_model_conv2_bias = quantized_model.conv2_bn_refused.layer.bias.detach().cpu()
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

            quantized_model_conv3_weight = quantized_model.conv3_bn_refused.layer.weight.detach().cpu()
            quantized_model_conv3_bias = quantized_model.conv3_bn_refused.layer.bias.detach().cpu()
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


class ValueSecondMomentTest(BaseSecondMomentTest):
    """
    This is the test for the Second Moment Correction feature.
    This test check that the gamma&beta values of the reconstructed BN didn't change during second moment application.
    """

    def create_feature_network(self, input_shape):
        return ConvSecondMomentNet()

    def run_test(self, seed=0):
        np.random.seed(seed)
        random.seed(a=seed)
        torch.random.manual_seed(seed)
        fw_info = DEFAULT_PYTORCH_INFO
        pytorch_impl = PytorchImplementation()
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            yield x

        model_float = self.create_feature_network(input_shapes)
        core_config_dict = self.get_core_configs()
        tpc_dict = self.get_tpc()

        for model_name, core_config in core_config_dict.items():
            tpc = tpc_dict[model_name]
            tg, graph_after_second_moment_correction = self.prepare_graph(model_float,
                                                                          representative_data_gen,
                                                                          core_config=core_config,
                                                                          fw_info=DEFAULT_PYTORCH_INFO,
                                                                          target_platform_capabilities=tpc)
            for node in graph_after_second_moment_correction.nodes:
                if node.layer_class == torch.nn.BatchNorm2d:
                    bf_second_moment_node = tg.find_node_by_name(node.name)[0]

                    gamma0 = bf_second_moment_node.get_weights_by_keys(GAMMA)
                    beta0 = bf_second_moment_node.get_weights_by_keys(BETA)
                    moving_mean0 = bf_second_moment_node.get_weights_by_keys(MOVING_MEAN)
                    moving_variance0 = bf_second_moment_node.get_weights_by_keys(MOVING_VARIANCE)

                    gamma1 = node.get_weights_by_keys(GAMMA)
                    beta1 = node.get_weights_by_keys(BETA)
                    moving_mean1 = node.get_weights_by_keys(MOVING_MEAN)
                    moving_variance1 = node.get_weights_by_keys(MOVING_VARIANCE)

                    # check that gamma&beta didn't change
                    self.unit_test.assertTrue(gamma0 == gamma1)
                    self.unit_test.assertTrue(beta0 == beta1)

                    # check that moving_mean&moving_variance did change
                    self.unit_test.assertFalse(moving_mean0 == moving_mean1)
                    self.unit_test.assertFalse(moving_variance0 == moving_variance1)

    def prepare_graph(self,
                      in_model: Module,
                      representative_data_gen: Callable,
                      core_config: CoreConfig = CoreConfig(),
                      fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                      target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_PYTORCH_INFO) -> \
            Tuple[Graph, Graph]:

        tb_w = _init_tensorboard_writer(fw_info)

        fw_impl = PytorchImplementation()

        tg, bit_widths_config = core_runner(in_model=in_model,
                                            representative_data_gen=representative_data_gen,
                                            core_config=core_config,
                                            fw_info=fw_info,
                                            fw_impl=fw_impl,
                                            tpc=target_platform_capabilities,
                                            tb_w=tb_w)
        graph_to_apply_second_moment = copy.deepcopy(tg)
        semi_quantized_model = quantized_model_builder_for_second_moment_correction(graph_to_apply_second_moment,
                                                                                    fw_info, fw_impl)
        pytorch_apply_second_moment_correction(semi_quantized_model, core_config, representative_data_gen,
                                               graph_to_apply_second_moment)

        return tg, graph_to_apply_second_moment
