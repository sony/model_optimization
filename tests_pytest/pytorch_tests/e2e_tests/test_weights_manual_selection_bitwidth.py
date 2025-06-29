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
import pytest

import model_compression_toolkit as mct
import torch
from torch.nn import Conv2d
from torch import add, sub

from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.target_platform_capabilities.constants import BIAS, PYTORCH_KERNEL, POSITIONAL_ATTR
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, WEIGHTS_N_BITS
from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core import CoreConfig
from mct_quantizers import PytorchQuantizationWrapper

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness, \
    AttributeQuantizationConfig
from mct_quantizers import QuantizationMethod

from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import BaseTorchIntegrationTest

kernel_weights_n_bits = 8
bias_weights_n_bits = 12
activation_n_bits = 8
default_weights_n_bits = 8


def get_op_qco(mp_kernel_n_bits, mp_bias_n_bits, quantized_bias=False, quantize_default_weight=False):
    # define a default quantization config for all non-specified weights attributes.
    default_weight_attr_config = AttributeQuantizationConfig(enable_weights_quantization=quantize_default_weight,
                                                             weights_n_bits=default_weights_n_bits)

    # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
    kernel_base_config = AttributeQuantizationConfig(
        weights_n_bits=kernel_weights_n_bits,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True)

    attr_weights_configs = {KERNEL_ATTR: kernel_base_config}
    if quantized_bias:
        attr_weights_configs[BIAS] = AttributeQuantizationConfig(weights_n_bits=bias_weights_n_bits,
                                                                 enable_weights_quantization=True)

    base_cfg = schema.OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping=attr_weights_configs,
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)

    default_config = schema.OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO
    )

    mx_cfg_list = [base_cfg]
    if mp_kernel_n_bits is not None:
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: mp_kernel_n_bits}}))
    if mp_bias_n_bits is not None:
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={BIAS: {WEIGHTS_N_BITS: mp_bias_n_bits}}))
    if mp_kernel_n_bits and mp_bias_n_bits:
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: mp_kernel_n_bits},
                                                                 BIAS: {WEIGHTS_N_BITS: mp_bias_n_bits}}))
    return base_cfg, mx_cfg_list, default_config


def generate_tpc_local(default_config, base_config, mixed_precision_cfg_list):
    default_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple([default_config]))
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple(mixed_precision_cfg_list),
        base_config=base_config)

    conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
    relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU)
    add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD)
    operator_set = [conv, relu, add]

    generated_tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        operator_set=tuple(operator_set))

    return generated_tpc


def generate_tpc_pos_attr_local(default_config):
    default_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple([default_config]))

    const_config_input16 = default_config.clone_and_edit(
        supported_input_activation_n_bits=(8, 16))

    # define a quantization config to quantize the positional weights into 16 bit (for layers where there is a
    # positional weight attribute).
    positional_weight_16_attr_config = schema.AttributeQuantizationConfig(
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_n_bits=16,
        weights_per_channel_threshold=False,
        enable_weights_quantization=True,
        lut_values_bitwidth=None)

    # define a quantization config to quantize the positional weights into 8 bit (for layers where there is a
    # positional weight attribute).
    positional_weight_8_attr_config = schema.AttributeQuantizationConfig(
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_n_bits=8,
        weights_per_channel_threshold=False,
        enable_weights_quantization=True,
        lut_values_bitwidth=None)

    const_config_input16_positional_weight16 = const_config_input16.clone_and_edit(
        attr_weights_configs_mapping={POSITIONAL_ATTR: positional_weight_16_attr_config})

    const_config_input16_positional_weight8 = const_config_input16.clone_and_edit(
        attr_weights_configs_mapping={POSITIONAL_ATTR: positional_weight_8_attr_config})
    const_configuration_options_inout16 = (
        schema.QuantizationConfigOptions(quantization_configurations=tuple([
            const_config_input16_positional_weight8,
            const_config_input16_positional_weight16]),
            base_config=const_config_input16_positional_weight8))

    # define a quantization config to quantize the positional weights into 2 bit (for layers where there is a
    # positional weight attribute).
    positional_weight_2_attr_config = schema.AttributeQuantizationConfig(
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_n_bits=2,
        weights_per_channel_threshold=False,
        enable_weights_quantization=True,
        lut_values_bitwidth=None)

    const_config_input16_positional_weight2 = const_config_input16.clone_and_edit(
        attr_weights_configs_mapping={POSITIONAL_ATTR: positional_weight_2_attr_config})
    const_configuration_options_inout_2 = (
        schema.QuantizationConfigOptions(quantization_configurations=tuple([
            const_config_input16,
            const_config_input16_positional_weight2]),
            base_config=const_config_input16))

    operator_set = []

    add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD, qc_options=const_configuration_options_inout16)
    sub = schema.OperatorsSet(name=schema.OperatorSetNames.SUB, qc_options=const_configuration_options_inout_2)
    operator_set.extend([add, sub])

    generated_tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        operator_set=tuple(operator_set))

    return generated_tpc


def representative_data_gen(shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
    for _ in range(num_iter):
        yield [torch.randn(batch_size, *shape)] * num_inputs


def get_float_model():
    class BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.relu(x)
            return x

    return BaseModel()


def get_float_model_with_constants():
    class BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            a = torch.rand(8)
            b = torch.rand(8)
            self.a = to_torch_tensor(a)
            self.b = to_torch_tensor(b)

        def forward(self, x):
            x = torch.add(x, self.a)
            x = torch.sub(self.b, x)
            return x

    return BaseModel()


class TestManualWeightsBitwidthSelectionByLayerType(BaseTorchIntegrationTest):
    def get_tpc(self, kernel_n_bits, bias_n_bits):
        base_cfg, mx_cfg_list, default_config = get_op_qco(kernel_n_bits, bias_n_bits)
        tpc = generate_tpc_local(default_config, base_cfg, mx_cfg_list)
        return tpc

    # (LayerType, bit width, attribute, kernel_n_bits, bias_n_bits)
    test_input_1 = (NodeTypeFilter(Conv2d), 16, PYTORCH_KERNEL, 16, None)
    test_input_2 = (NodeTypeFilter(Conv2d), [2], [PYTORCH_KERNEL], 2, None)

    test_expected_1 = ([Conv2d], [16])
    test_expected_2 = ([Conv2d], [2])

    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
    ])
    def test_manual_weights_bitwidth_selection(self, inputs, expected):
        float_model = get_float_model()
        target_platform_cap = self.get_tpc(kernel_n_bits=inputs[3], bias_n_bits=inputs[4])

        core_config = CoreConfig()
        core_config.bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])

        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=float_model,
            representative_data_gen=representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=target_platform_cap
        )
        wrappers = self.fetch_model_layers_by_cls(quantized_model, PytorchQuantizationWrapper)
        assert len(wrappers) == len([m for m in float_model.modules() if isinstance(m, Conv2d)]) == 2
        expected_bitwidths = expected[1]
        attrs = inputs[2] if isinstance(inputs[2], list) else [inputs[2]]
        for wrapper in wrappers:
            for attr, exp_bw in zip(attrs, expected_bitwidths):
                assert self.fetch_weight_quantizer(wrapper, attr).num_bits == exp_bw


class TestManualWeightsBitwidthSelectionByLayerName(BaseTorchIntegrationTest):
    def get_float_model(self):
        return get_float_model()

    def get_tpc(self, kernel_n_bits, bias_n_bits, quantized_bias):
        base_cfg, mx_cfg_list, default_config = get_op_qco(kernel_n_bits, bias_n_bits, quantized_bias)
        tpc = generate_tpc_local(default_config, base_cfg, mx_cfg_list)
        return tpc

    # (LayerName, bit width, attribute, kernel_n_bits, bias_n_bits)
    test_input_1 = (NodeNameFilter("conv1"), 16, PYTORCH_KERNEL, 16, None)
    test_input_2 = (NodeNameFilter("conv1"), [2], [PYTORCH_KERNEL], 2, None)
    test_input_3 = ([NodeNameFilter("conv1"), NodeNameFilter("conv1")], [4, 16], [PYTORCH_KERNEL, BIAS], 4, 16)

    test_expected_1 = (["conv1"], [16])
    test_expected_2 = (["conv1"], [2])
    test_expected_3 = (["conv1", "conv1"], [4, 16])

    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
        (test_input_3, test_expected_3),
    ])
    def test_manual_weights_bitwidth_selection(self, inputs, expected):

        float_model = self.get_float_model()

        target_platform_cap = self.get_tpc(kernel_n_bits=inputs[3], bias_n_bits=inputs[4], quantized_bias=True)

        core_config = CoreConfig()
        core_config.bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])

        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=float_model,
            representative_data_gen=representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=target_platform_cap
        )
        wrappers = self.fetch_model_layers_by_cls(quantized_model, PytorchQuantizationWrapper, named=True)
        assert len(wrappers) == len([m for m in float_model.modules() if isinstance(m, Conv2d)]) == 2
        bitwidths = expected[1]
        attrs = inputs[2] if isinstance(inputs[2], list) else [inputs[2]]
        for attr, exp_bw in zip(attrs, bitwidths):
            assert self.fetch_weight_quantizer(wrappers['conv1'], attr).num_bits == exp_bw
        if BIAS not in attrs:
            assert self.fetch_weight_quantizer(wrappers['conv1'], BIAS).num_bits == bias_weights_n_bits

        assert self.fetch_weight_quantizer(wrappers['conv2'], PYTORCH_KERNEL).num_bits == kernel_weights_n_bits
        assert self.fetch_weight_quantizer(wrappers['conv2'], BIAS).num_bits == bias_weights_n_bits


class TestManualPositionalAttrWeightsBitwidth(BaseTorchIntegrationTest):
    def get_tpc(self, kernel_n_bits, bias_n_bits):
        _, _, default_config = get_op_qco(kernel_n_bits, bias_n_bits, quantize_default_weight=True)
        tpc = generate_tpc_pos_attr_local(default_config)
        return tpc

    # (LayerType, bit width, attribute)
    test_input_1 = (NodeTypeFilter(add), 16, POSITIONAL_ATTR)
    test_input_2 = (NodeTypeFilter(sub), [2], [POSITIONAL_ATTR])
    test_input_3 = (NodeNameFilter('add'), 16, POSITIONAL_ATTR)
    test_input_4 = (NodeNameFilter('sub'), 2, [POSITIONAL_ATTR])

    @pytest.mark.parametrize(("inputs", "exp_bitwidth", "layer_name"), [
        (test_input_1, 16, 'add'),
        (test_input_2, 2, 'sub'),
        (test_input_3, 16, 'add'),
        (test_input_4, 2, 'sub')
    ])
    def test_manual_weights_bitwidth_selection(self, inputs, exp_bitwidth, layer_name):
        float_model = get_float_model_with_constants()
        target_platform_cap = self.get_tpc(None, None)

        core_config = CoreConfig()
        core_config.bit_width_config.set_manual_weights_bit_width(*inputs)

        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=float_model,
            representative_data_gen=representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=target_platform_cap
        )
        wrappers = self.fetch_model_layers_by_cls(quantized_model, PytorchQuantizationWrapper, named=True)
        assert len(wrappers) == 2
        pos_ind = {'add': 1, 'sub': 0}
        for name, wrapper in wrappers.items():
            if name == layer_name:
                assert self.fetch_weight_quantizer(wrappers[layer_name], pos_ind[layer_name]).num_bits == exp_bitwidth
            else:
                assert self.fetch_weight_quantizer(wrappers[name], 1-pos_ind[layer_name]).num_bits == 8
