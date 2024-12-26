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
from model_compression_toolkit.core import MixedPrecisionQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorSetNames
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import get_opset_by_name
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.attach2pytorch import \
    AttachTpModelToPytorch
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4.tp_model import get_tp_model
from model_compression_toolkit.core.pytorch.utils import get_working_device
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest


get_op_set = lambda x, x_list: [op_set for op_set in x_list if op_set.name == x][0]


class Activation16BitNet(torch.nn.Module):

    def __init__(self, use_concat=True, enable_head=True):
        super().__init__()
        self.use_concat = use_concat
        self.enable_head = enable_head
        self.conv = torch.nn.Conv2d(3, 3, 1)
        if enable_head:
            self.conv_a = torch.nn.Conv2d(3, 3, 1)
            self.conv_b = torch.nn.Conv2d(3, 3, 1)
        self.register_buffer('add_const', torch.rand((3, 1, 1)))
        self.register_buffer('sub_const', torch.rand((3, 1, 1)))
        self.register_buffer('div_const', 2*torch.ones((3, 1, 1)))

    def forward(self, x):
        x = torch.mul(x, x)
        if self.use_concat:
            x = torch.concat([x, x], dim=2)
        x1 = torch.add(x, self.add_const)
        x = torch.sub(x, self.sub_const)
        x = torch.mul(x, x1)
        x = torch.reshape(x, (-1, 3, 2*(1+int(self.use_concat)), 4, 8))
        x = torch.reshape(x, (-1, 3, 8*(1+int(self.use_concat)), 8))
        x = self.conv(x)
        x = torch.divide(x, self.div_const)

        if self.enable_head:
            x = torch.cat([torch.nn.functional.gelu(self.conv_a(x)),
                           torch.nn.functional.tanh(self.conv_b(x))], dim=1)

        return x


class Activation16BitNetMP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('add_const', torch.rand((3, 1, 1)))
        self.register_buffer('sub_const', torch.rand((3, 1, 1)))
        self.register_buffer('div_const', 2*torch.ones((3, 1, 1)))

    def forward(self, x):
        x = torch.mul(x, x)[:, :, :8, :8]
        x1 = torch.add(x, self.add_const)
        x = torch.sub(x, self.sub_const)
        x = torch.mul(x, x1)
        x = torch.reshape(x, (-1, 3, 2, 4, 8))
        x = torch.reshape(x, (-1, 3, 8, 8))
        x = torch.divide(x, self.div_const)

        return x


def set_16bit_as_default(tpc, required_op_set):
    # base_config = [l for l in tpc.layer2qco[op].quantization_configurations if l.activation_n_bits == 16][0]
    # tpc.layer2qco[op] = replace(tpc.layer2qco[op], base_config=base_config)
    pass


class Activation16BitTest(BasePytorchFeatureNetworkTest):

    def get_tpc(self):
        # TODO: need to build a TP model that puts the 16 bit oprion ad default, but include also all other configs, opsets and fusing as in the v4 tpc
        raise Exception(
            "TODO: need to build a TP model that puts the 16 bit oprion ad default, but include also all other configs, opsets and fusing as in the v4 tpc")
        tpc = get_tp_model()
        set_16bit_as_default(tpc, OPSET_MUL, [torch.mul, mul])
        set_16bit_as_default(tpc, OPSET_GELU, [torch.nn.GELU, torch.nn.functional.gelu])
        set_16bit_as_default(tpc, OPSET_TANH, [torch.nn.Tanh, torch.nn.functional.tanh, torch.tanh])
        return tpc

    def create_networks(self):
        return Activation16BitNet()

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        x = torch.from_numpy(input_x[0].astype('float32')).to(get_working_device())
        out_f = float_model.to(get_working_device())(x)
        out_q = quantized_model(x)
        self.unit_test.assertTrue(out_f.shape == out_q.shape, "Output shape mismatch.")

        mul1_act_quant = quantized_model.mul_activation_holder_quantizer
        mul2_act_quant = quantized_model.mul_1_activation_holder_quantizer
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.num_bits == 16,
                                  "1st mul activation bits should be 16 bits because of following concat node.")
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.signed == True,
                                  "1st mul activation should be forced by TPC to be signed, even though activations as all positive.")
        self.unit_test.assertTrue(mul2_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "2nd mul activation bits should be 8 bits because of following div node.")
        self.unit_test.assertTrue(quantized_model.gelu_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16,
                                  "gelu activation bits should be 16 bits because of following concat node.")
        self.unit_test.assertTrue(quantized_model.tanh_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16,
                                  "tanh activation bits should be 16 bits because of following concat node.")


class Activation16BitMixedPrecisionTest(Activation16BitTest):

    def get_tpc(self):
        tpc = mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, 'v4')
        mul_op_set = get_op_set('Mul', tpc.tp_model.operator_set)
        base_config = [l for l in mul_op_set.qc_options.quantization_configurations if l.activation_n_bits == 16][0]
        quantization_configurations = list(mul_op_set.qc_options.quantization_configurations)
        quantization_configurations.extend([
            tpc.layer2qco[torch.mul].base_config.clone_and_edit(activation_n_bits=4),
            tpc.layer2qco[torch.mul].base_config.clone_and_edit(activation_n_bits=2)])
        tpc.layer2qco[torch.mul] = tpc.layer2qco[torch.mul].model_copy(
            update={'base_config': base_config, 'quantization_configurations': tuple(quantization_configurations)})
        tpc.layer2qco[mul] = tpc.layer2qco[mul].model_copy(
            update={'base_config': base_config, 'quantization_configurations': tuple(quantization_configurations)})
        return tpc

    def get_resource_utilization(self):
        return mct.core.ResourceUtilization(activation_memory=5000)

    def create_networks(self):
        return Activation16BitNetMP()

    def get_mixed_precision_config(self):
        return MixedPrecisionQuantizationConfig()

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        mul1_act_quant = quantized_model.mul_activation_holder_quantizer
        mul2_act_quant = quantized_model.mul_1_activation_holder_quantizer
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "1st mul activation bits should be 8 bits because of RU.")
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.signed == False,
                                  "1st mul activation should be unsigned, because activations as all positive.")
        self.unit_test.assertTrue(mul2_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "2nd mul activation bits should be 8 bits because of following div node.")
