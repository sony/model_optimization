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

import inspect

from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from mct_quantizers import PytorchActivationQuantizationHolder
import model_compression_toolkit as mct
import torch
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_activation_test import \
    MixedPrecisionActivationBaseTest
from tests.pytorch_tests.utils import get_layer_type_from_activation_quantizer

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


class NetForBitSelection(torch.nn.Module):
    def __init__(self, input_shape):
        super(NetForBitSelection, self).__init__()
        b, in_channels, h, w = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.bn2 = torch.nn.BatchNorm2d(in_channels)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(in_channels * h * w, 5)

    def forward(self, inp):
        x = self.conv1(inp)
        out1 = self.bn1(x)
        x = out1 + 3
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x + out1
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


class BaseManualBitWidthSelectionTest(MixedPrecisionActivationBaseTest):

    def create_feature_network(self, input_shape):
        return NetForBitSelection(input_shape)

    def get_mp_core_config(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False)
        mpc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

        core_config = mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)
        return core_config

    def get_core_configs(self):
        # Configures the core settings including manual bit width adjustments.
        core_config = self.get_mp_core_config()
        core_config.bit_width_config.set_manual_activation_bit_width(self.filters, self.bit_widths)
        return {"mixed_precision_activation_model": core_config}

class ManualBitWidthByLayerTypeTest(BaseManualBitWidthSelectionTest):
    """
        This test check the manual bit width configuration.
        Call it with a layer type filter or list of layer type filters, bit width or list of bit widths.
        Uses the manual bit width API in the "get_core_configs" method.
        """
    def __init__(self, unit_test, filters, bit_widths):
        self.filters = filters
        self.bit_widths = bit_widths
        self.layer_types = {}
        self.functional_names = {}

        filters = [filters] if not isinstance(filters, list) else filters
        bit_widths = [bit_widths] if not isinstance(bit_widths, list) else bit_widths
        if len(bit_widths) < len(filters):
            bit_widths = [bit_widths[0] for f in filters]
        for filter, bit_width in zip(filters, bit_widths):
            if inspect.isclass(filter.node_type):
                self.layer_types.update({filter.node_type: bit_width})
            else:
                self.functional_names.update({filter.node_type.__name__: bit_width})

        super().__init__(unit_test)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        # in the compare we need bit_widths to be a list
        bit_widths = [self.bit_widths] if not isinstance(self.bit_widths, list) else self.bit_widths

        for model_name, quantized_model in quantized_models.items():
            for name, layer in quantized_model.named_modules():
                # check if the layer is an activation quantizer
                if isinstance(layer, PytorchActivationQuantizationHolder):
                    # get the layer that's activation is being quantized
                    layer_type = get_layer_type_from_activation_quantizer(quantized_model, name)

                    # check if this layer is in the layer types to change bit width and check that the correct bit width was applied.
                    if self.layer_types.get(type(layer_type)) is not None:
                        self.unit_test.assertTrue(layer.activation_holder_quantizer.num_bits == self.layer_types.get(type(layer_type)))
                    # else check if this layer is in the functional types to change bit width and check that the correct bit width was applied.
                    elif any([k in name for k in self.functional_names.keys()]):
                        for k in self.functional_names.keys():
                            if k in name:
                                bit_width = self.functional_names.get(k)
                        self.unit_test.assertTrue(layer.activation_holder_quantizer.num_bits == bit_width)
                    else:
                        # make sure that the bit width of other layers was not changed.
                        self.unit_test.assertFalse(layer.activation_holder_quantizer.num_bits in bit_widths, msg=f"name {name}, layer.activation_holder_quantizer.num_bits {layer.activation_holder_quantizer.num_bits }, {self.bit_widths}")


class ManualBitWidthByLayerNameTest(BaseManualBitWidthSelectionTest):
    """
    This test check the manual bit width configuration.
    Call it with a name filter or list of name filters, bit width or list of bit widths.
    Uses the manual bit width API in the "get_core_configs" method.
    """
    def __init__(self, unit_test, filters, bit_widths):
        self.filters = filters
        self.bit_widths = bit_widths
        self.layer_names = {}

        filters = [filters] if not isinstance(filters, list) else filters
        bit_widths = [bit_widths] if not isinstance(bit_widths, list) else bit_widths
        if len(bit_widths) < len(filters):
            bit_widths = [bit_widths[0] for f in filters]
        for filter, bit_width in zip(filters, bit_widths):
            self.layer_names.update({filter.node_name: bit_width})


        super().__init__(unit_test)


    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        # in the compare we need bit_widths to be a list
        bit_widths = [self.bit_widths] if not isinstance(self.bit_widths, list) else self.bit_widths

        for model_name, quantized_model in quantized_models.items():
            for name, layer in quantized_model.named_modules():
                if isinstance(layer, PytorchActivationQuantizationHolder):
                    # check if this layer is in the layer names to change bit width and check that the correct bit width was applied.
                    if any([layer_name == name.split("_activation")[0] for layer_name in self.layer_names.keys()]):
                        for layer_name, bit_width in self.layer_names.items():
                            if layer_name == name.split("_activation")[0]:
                                break
                        self.unit_test.assertTrue(layer.activation_holder_quantizer.num_bits == bit_width)
                    else:
                        # make sure that the bit width of other layers was not changed.
                        self.unit_test.assertFalse(layer.activation_holder_quantizer.num_bits in bit_widths, msg=f"name {name}, layer.activation_holder_quantizer.num_bits {layer.activation_holder_quantizer.num_bits }, {self.bit_widths}")


class Manual16BitTest(ManualBitWidthByLayerNameTest):

    def get_tpc(self):
        tpc = mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, 'v3')
        mul_op_set = get_op_set('Mul', tpc.tp_model.operator_set)
        mul_op_set.qc_options.base_config = [l for l in mul_op_set.qc_options.quantization_config_list if l.activation_n_bits == 16][0]
        tpc.layer2qco[torch.mul].base_config = mul_op_set.qc_options.base_config
        tpc.layer2qco[mul].base_config = mul_op_set.qc_options.base_config
        return {'mixed_precision_activation_model': tpc}

    def create_feature_network(self, input_shape):
        return Activation16BitNet()


class Manual16BitTestMixedPrecisionTest(ManualBitWidthByLayerNameTest):

    def get_tpc(self):
        tpc = mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, 'v3')
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

        return {'mixed_precision_activation_model': tpc}

    def get_resource_utilization(self):
        return mct.core.ResourceUtilization(activation_memory=6200)


    def create_feature_network(self, input_shape):
        return Activation16BitNet()