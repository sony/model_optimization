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

import inspect

from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder
from mct_quantizers import PytorchQuantizationWrapper
import model_compression_toolkit as mct
import torch

from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers

from tests.common_tests.helpers.generate_test_tpc import generate_tpc_with_activation_mp
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_activation_test import \
    MixedPrecisionActivationBaseTest
from tests.pytorch_tests.utils import get_layer_type_from_activation_quantizer, extract_model_weights
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs

from tests.pytorch_tests.tpc_pytorch import get_mp_activation_pytorch_tpc_dict
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR

get_op_set = lambda x, x_list: [op_set for op_set in x_list if op_set.name == x][0]


class Weights16BitNet(torch.nn.Module):

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


class BaseManualWeightsBitWidthSelectionTest(MixedPrecisionActivationBaseTest):

    def create_feature_network(self, input_shape):
        return NetForBitSelection(input_shape)

    def get_tpc(self):
        mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]
        kernel_weights_n_bits = 16
        bias_weights_n_bits = 16
        activation_n_bits = 8

        base_cfg, _, default_config = get_op_quantization_configs()
        base_config = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                                {
                                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                                .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                                .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                                },
                                                activation_n_bits=activation_n_bits)

        tpc_dict = get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=mp_bitwidth_candidates_list),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

        return {"manual_bit_selection": v for _, v in tpc_dict.items()}
   
    @staticmethod
    def get_mp_core_config():
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False,
                                         custom_tpc_opset_to_layer={"Input": CustomOpsetLayers([DummyPlaceHolder])})
        mpc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

        core_config = mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)
        return core_config

    def get_core_configs(self):
        # Configures the core settings including manual bit width adjustments.
        core_config = self.get_mp_core_config()
        core_config.bit_width_config.set_manual_weights_bit_width(self.filters, self.bit_widths, self.attrs)
        return {"manual_bit_selection": core_config}


class ManualWeightsBitWidthByLayerTypeTest(BaseManualWeightsBitWidthSelectionTest):
    """
    This test check the manual weights bit width configuration.
    Call it with a layer type filter or list of layer type filters, bit width or list of bit widths,
    attribute or list of attribute.
    Uses the manual bit width API in the "get_core_configs" method.
    """
    def __init__(self, unit_test, filters, bit_widths, attrs):
        self.filters = filters
        self.bit_widths = bit_widths
        self.attrs = attrs
        self.layer_types = []
        self.functional_names = []

        filters = [filters] if not isinstance(filters, list) else filters
        bit_widths = [bit_widths] if not isinstance(bit_widths, list) else bit_widths
        attrs = [attrs] if not isinstance(attrs, list) else attrs
        if len(bit_widths) < len(filters):
            bit_widths = [bit_widths[0] for f in filters]
        for filter, bit_width, attr in zip(filters, bit_widths, attrs):
            if inspect.isclass(filter.node_type):
                self.layer_types.append([filter.node_type, [bit_width, attr]])
            else:
                self.functional_names.append([filter.node_type.__name__, [bit_width, attr]])

        super().__init__(unit_test)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        # in the compare we need bit_widths to be a list
        bit_widths = [self.bit_widths] if not isinstance(self.bit_widths, list) else self.bit_widths

        for model_name, quantized_model in quantized_models.items():
            for name, layer in quantized_model.named_modules():
                # check if the layer is a weights quantizer
                if isinstance(layer, PytorchQuantizationWrapper):

                    # check if this layer is in the layer types to change bit width and check that the correct bit width was applied.
                    bitwidth_attrs = []
                    for layer_type in self.layer_types:
                        if layer_type[0] == type(layer.layer):
                            bitwidth_attrs.append(layer_type[1])

                    for bitwidth_attr in bitwidth_attrs:
                        bitwidth = bitwidth_attr[0]
                        attr = bitwidth_attr[1]
                        if layer.weights_quantizers.get(attr) is not None:
                            if bitwidth is not None:
                                self.unit_test.assertTrue(layer.weights_quantizers.get(attr).num_bits == bitwidth)
                        else:
                            # make sure that the bit width of other layers was not changed.
                            self.unit_test.assertFalse(layer.weights_quantizers.get(attr).num_bits in bit_widths, 
                                                       msg=f"name {name}, quantizer.num_bits {layer.weights_quantizers.get(attr).num_bits }, {self.bit_widths}")


class ManualWeightsBias2BitWidthByLayerTypeTest(ManualWeightsBitWidthByLayerTypeTest):
    """
    This test check the 2 bit manual weights bit width configuration of bias by layer type.
    """
    def get_tpc(self):
        mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]
        kernel_weights_n_bits = 16
        bias_weights_n_bits = 2
        activation_n_bits = 8

        base_cfg, _, default_config = get_op_quantization_configs()
        base_config = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                                {
                                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                                .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                                .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                                },
                                                activation_n_bits=activation_n_bits)

        tpc_dict = get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=mp_bitwidth_candidates_list),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

        return {"manual_bit_selection": v for _, v in tpc_dict.items()}


class ManualWeightsBias4BitWidthByLayerTypeTest(ManualWeightsBitWidthByLayerTypeTest):
    """
    This test check the 4 bit manual weights bit width configuration of bias by layer type.
    """
    def get_tpc(self):
        mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]
        kernel_weights_n_bits = 16
        bias_weights_n_bits = 4
        activation_n_bits = 8

        base_cfg, _, default_config = get_op_quantization_configs()
        base_config = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                                {
                                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                                .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                                .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                                },
                                                activation_n_bits=activation_n_bits)

        tpc_dict = get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=mp_bitwidth_candidates_list),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

        return {"manual_bit_selection": v for _, v in tpc_dict.items()}


class ManualWeightsBias8BitWidthByLayerTypeTest(ManualWeightsBitWidthByLayerTypeTest):
    """
    This test check the 8 bit manual weights bit width configuration of bias by layer type.
    """
    def get_tpc(self):
        mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]
        kernel_weights_n_bits = 16
        bias_weights_n_bits = 8
        activation_n_bits = 8

        base_cfg, _, default_config = get_op_quantization_configs()
        base_config = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                                {
                                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                                .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                                .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                                },
                                                activation_n_bits=activation_n_bits)

        tpc_dict = get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=mp_bitwidth_candidates_list),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

        return {"manual_bit_selection": v for _, v in tpc_dict.items()}


class ManualWeightsBias32BitWidthByLayerTypeTest(ManualWeightsBitWidthByLayerTypeTest):
    """
    This test check the 32 bit manual weights bit width configuration of bias by layer type.
    """
    def get_tpc(self):
        mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]
        kernel_weights_n_bits = 16
        bias_weights_n_bits = 32
        activation_n_bits = 8

        base_cfg, _, default_config = get_op_quantization_configs()
        base_config = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                                {
                                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                                .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                                .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                                },
                                                activation_n_bits=activation_n_bits)

        tpc_dict = get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=mp_bitwidth_candidates_list),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

        return {"manual_bit_selection": v for _, v in tpc_dict.items()}


class ManualWeightsBitWidthByLayerNameTest(BaseManualWeightsBitWidthSelectionTest):
    """
    This test check the manual weights bit width configuration.
    Call it with a layer name filter or list of layer name filters, bit width or list of bit widths,
    attribute or list of attribute.
    Uses the manual bit width API in the "get_core_configs" method.
    """
    def __init__(self, unit_test, filters, bit_widths, attrs):
        self.filters = filters
        self.bit_widths = bit_widths
        self.attrs = attrs
        self.layer_names = []

        filters = [filters] if not isinstance(filters, list) else filters
        bit_widths = [bit_widths] if not isinstance(bit_widths, list) else bit_widths
        attrs = [attrs] if not isinstance(attrs, list) else attrs
        if len(bit_widths) < len(filters):
            bit_widths = [bit_widths[0] for f in filters]
        if len(attrs) < len(attrs):
            attrs = [attrs[0] for f in attrs]
        for filter, bit_width, attr in zip(filters, bit_widths, attrs):
            self.layer_names.append([filter.node_name, [bit_width, attr]])

        super().__init__(unit_test)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        # in the compare we need bit_widths to be a list
        bit_widths = [self.bit_widths] if not isinstance(self.bit_widths, list) else self.bit_widths

        for model_name, quantized_model in quantized_models.items():
            for name, layer in quantized_model.named_modules():
                # check if the layer is a weights quantizer
                if isinstance(layer, PytorchQuantizationWrapper):
                    # check if this layer is in the layer types to change bit width and check that the correct bit width was applied.
                    bitwidth_attrs = []
                    for layer_name in self.layer_names:
                        if layer_name[0] == name:
                            bitwidth_attrs.append(layer_name[1])

                    for bitwidth_attr in bitwidth_attrs:
                        bitwidth = bitwidth_attr[0]
                        attr = bitwidth_attr[1]
                        if layer.weights_quantizers.get(attr) is not None:
                            if bitwidth is not None:
                                self.unit_test.assertTrue(layer.weights_quantizers.get(attr).num_bits == bitwidth)
                        else:
                            # make sure that the bit width of other layers was not changed.
                            self.unit_test.assertFalse(layer.weights_quantizers.get(attr).num_bits in bit_widths, 
                                                       msg=f"name {name}, quantizer.num_bits {layer.weights_quantizers.get(attr).num_bits }, {self.bit_widths}")


class ManualWeightsBias2BitWidthByLayerNameTest(ManualWeightsBitWidthByLayerNameTest):
    """
    This test check the 2 bit manual weights bit width configuration of bias by layer name.
    """
    def get_tpc(self):
        mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]
        kernel_weights_n_bits = 16
        bias_weights_n_bits = 2
        activation_n_bits = 8

        base_cfg, _, default_config = get_op_quantization_configs()
        base_config = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                                {
                                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                                .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                                .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                                },
                                                activation_n_bits=activation_n_bits)

        tpc_dict = get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=mp_bitwidth_candidates_list),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

        return {"manual_bit_selection": v for _, v in tpc_dict.items()}


class ManualWeightsBias4BitWidthByLayerNameTest(ManualWeightsBitWidthByLayerNameTest):
    """
    This test check the 4 bit manual weights bit width configuration of bias by layer name.
    """
    def get_tpc(self):
        mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]
        kernel_weights_n_bits = 16
        bias_weights_n_bits = 4
        activation_n_bits = 8

        base_cfg, _, default_config = get_op_quantization_configs()
        base_config = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                                {
                                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                                .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                                .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                                },
                                                activation_n_bits=activation_n_bits)

        tpc_dict = get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=mp_bitwidth_candidates_list),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

        return {"manual_bit_selection": v for _, v in tpc_dict.items()}


class ManualWeightsBias8BitWidthByLayerNameTest(ManualWeightsBitWidthByLayerNameTest):
    """
    This test check the 8 bit manual weights bit width configuration of bias by layer name.
    """
    def get_tpc(self):
        mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]
        kernel_weights_n_bits = 16
        bias_weights_n_bits = 8
        activation_n_bits = 8

        base_cfg, _, default_config = get_op_quantization_configs()
        base_config = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                                {
                                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                                .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                                .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                                },
                                                activation_n_bits=activation_n_bits)

        tpc_dict = get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=mp_bitwidth_candidates_list),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

        return {"manual_bit_selection": v for _, v in tpc_dict.items()}


class ManualWeightsBias32BitWidthByLayerNameTest(ManualWeightsBitWidthByLayerNameTest):
    """
    This test check the 32 bit manual weights bit width configuration of bias by layer name.
    """
    def get_tpc(self):
        mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]
        kernel_weights_n_bits = 16
        bias_weights_n_bits = 32
        activation_n_bits = 8

        base_cfg, _, default_config = get_op_quantization_configs()
        base_config = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                                {
                                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                                .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                                .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                                },
                                                activation_n_bits=activation_n_bits)

        tpc_dict = get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=mp_bitwidth_candidates_list),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

        return {"manual_bit_selection": v for _, v in tpc_dict.items()}
