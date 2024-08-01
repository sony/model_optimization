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
from operator import mul
import torch

import model_compression_toolkit as mct
from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest


get_op_set = lambda x, x_list: [op_set for op_set in x_list if op_set.name == x][0]


class Activation16BitNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.register_buffer('add_const', torch.rand((3, 1, 1)))
        self.register_buffer('sub_const', torch.rand((3, 1, 1)))
        self.register_buffer('div_const', 2*torch.ones((3, 1, 1)))

    def forward(self, x):
        x = torch.mul(x, x)
        x1 = torch.add(x, self.add_const)
        x = torch.sub(x, self.sub_const)
        x = torch.mul(x, x1)
        x = self.conv(x)
        x = torch.divide(x, self.div_const)
        return x


class Activation16BitTest(BasePytorchFeatureNetworkTest):

    def get_tpc(self):
        tpc = mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, 'v4')
        mul_op_set = get_op_set('Mul', tpc.tp_model.operator_set)
        mul_op_set.qc_options.base_config = [l for l in mul_op_set.qc_options.quantization_config_list if l.activation_n_bits == 16][0]
        tpc.layer2qco[torch.mul].base_config = mul_op_set.qc_options.base_config
        tpc.layer2qco[mul].base_config = mul_op_set.qc_options.base_config
        return tpc

    def create_networks(self):
        return Activation16BitNet()

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        mul1_act_quant = quantized_model.mul_activation_holder_quantizer
        mul2_act_quant = quantized_model.mul_1_activation_holder_quantizer
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.num_bits == 16,
                                  "1st mul activation bits should be 16 bits because of following add node.")
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.signed == True,
                                  "1st mul activation should be forced by TPC to be signed, even though activations as all positive.")
        self.unit_test.assertTrue(mul2_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "2nd mul activation bits should be 8 bits because of following div node.")


class Activation16BitMixedPrecisionTest(Activation16BitTest):

    def get_tpc(self):
        tpc = mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, 'v4')
        mul_op_set = get_op_set('Mul', tpc.tp_model.operator_set)
        mul_op_set.qc_options.base_config = [l for l in mul_op_set.qc_options.quantization_config_list if l.activation_n_bits == 16][0]
        tpc.layer2qco[torch.mul].base_config = mul_op_set.qc_options.base_config
        tpc.layer2qco[mul].base_config = mul_op_set.qc_options.base_config
        mul_op_set.qc_options.quantization_config_list.extend(
            [mul_op_set.qc_options.base_config.clone_and_edit(activation_n_bits=4),
             mul_op_set.qc_options.base_config.clone_and_edit(activation_n_bits=2)])
        tpc.layer2qco[torch.mul].quantization_config_list.extend([
            tpc.layer2qco[torch.mul].base_config.clone_and_edit(activation_n_bits=4),
            tpc.layer2qco[torch.mul].base_config.clone_and_edit(activation_n_bits=2)])
        tpc.layer2qco[mul].quantization_config_list.extend([
            tpc.layer2qco[mul].base_config.clone_and_edit(activation_n_bits=4),
            tpc.layer2qco[mul].base_config.clone_and_edit(activation_n_bits=2)])

        return tpc

    def get_resource_utilization(self):
        return mct.core.ResourceUtilization(activation_memory=200)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        mul1_act_quant = quantized_model.mul_activation_holder_quantizer
        mul2_act_quant = quantized_model.mul_1_activation_holder_quantizer
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "1st mul activation bits should be 8 bits because of RU.")
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.signed == False,
                                  "1st mul activation should be unsigned, because activations as all positive.")
        self.unit_test.assertTrue(mul2_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "2nd mul activation bits should be 8 bits because of following div node.")
