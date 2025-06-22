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

from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core import BitWidthConfig, QuantizationConfig

from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph

import torch
from torch import nn

from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness, \
    AttributeQuantizationConfig
from mct_quantizers import QuantizationMethod

from model_compression_toolkit.core.pytorch.default_framework_info import PyTorchInfo
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation

from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, WEIGHTS_N_BITS, POS_ATTR
from model_compression_toolkit.target_platform_capabilities.constants import PYTORCH_KERNEL


class TestManualWeightsBitwidthSelection:
    def get_op_qco(self):
        # define a default quantization config for all non-specified weights attributes.
        default_weight_attr_config = AttributeQuantizationConfig()

        # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
        kernel_base_config = AttributeQuantizationConfig(
            weights_n_bits=8,
            weights_per_channel_threshold=True,
            enable_weights_quantization=True)

        base_cfg = schema.OpQuantizationConfig(
            default_weight_attr_config=default_weight_attr_config,
            attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config},
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
        for n in [2, 4, 16]:
            mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: n}}))

        return base_cfg, mx_cfg_list, default_config

    def generate_tpc_local(self, default_config, base_config, mixed_precision_cfg_list):
        default_configuration_options = schema.QuantizationConfigOptions(
            quantization_configurations=tuple([default_config]))
        mixed_precision_configuration_options = schema.QuantizationConfigOptions(
            quantization_configurations=tuple(mixed_precision_cfg_list),
            base_config=base_config)

        operator_set = []

        conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
        relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU)
        add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD)
        operator_set.extend([conv, relu, add])

        generated_tpc = schema.TargetPlatformCapabilities(
            default_qco=default_configuration_options,
            operator_set=tuple(operator_set))

        return generated_tpc

    def get_tpc(self):
        base_cfg, mx_cfg_list, default_config = self.get_op_qco()
        tpc = self.generate_tpc_local(default_config, base_cfg, mx_cfg_list)
        return tpc

    def representative_data_gen(self, shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
        for _ in range(num_iter):
            yield [torch.randn(batch_size, *shape)] * num_inputs

    def get_float_model(self):
        class BaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = torch.add(x, 2)
                x = self.relu(x)
                return x

        return BaseModel()

    def get_test_graph(self, qc):
        float_model = self.get_float_model()
        fw_info = PyTorchInfo

        fw_impl = PytorchImplementation()
        graph = fw_impl.model_reader(float_model,
                                     self.representative_data_gen)

        tpc = self.get_tpc()
        attach2pytorch = AttachTpcToPytorch()
        fqc = attach2pytorch.attach(
            tpc, qc.custom_tpc_opset_to_layer)
        graph.set_fqc(fqc)

        return graph

    # test case for set_manual_activation_bit_width
    """
    Test Items Policy:
        - How to specify the target layer: Options(type/name)
        - Target attribute information: Options(kernel) 
        - Bit width variations: Options(2, 4, 16)
    """
    test_input_1 = (NodeNameFilter("conv1"), 2, PYTORCH_KERNEL)
    test_input_2 = (NodeTypeFilter(nn.Conv2d), 16, PYTORCH_KERNEL)
    test_input_3 = ([NodeNameFilter("conv1"), NodeNameFilter("conv2")], [4, 8], [PYTORCH_KERNEL, PYTORCH_KERNEL])
    test_input_4 = (NodeNameFilter("add"), 2, PYTORCH_KERNEL, False)
    test_input_5 = (NodeNameFilter("add"), 2, PYTORCH_KERNEL, True)

    test_expected_1 = ({"conv1": {PYTORCH_KERNEL: 2}})
    test_expected_2 = ({"conv1": {PYTORCH_KERNEL: 16}, "conv2": {PYTORCH_KERNEL: 16}})
    test_expected_3 = ({"conv1": {PYTORCH_KERNEL: 4}, "conv2": {PYTORCH_KERNEL: 8}})
    test_expected_4 = ('The requested attribute weight to change the bit width for add:add does not exist.')
    test_expected_5 = ('The requested attribute weight to change the bit width for add:add does not exist.')

    @pytest.mark.parametrize(
        ("inputs", "expected"), [
            (test_input_1, test_expected_1),
            (test_input_2, test_expected_2),
            (test_input_3, test_expected_3),
        ])
    def test_manual_weights_bitwidth_selection(self, inputs, expected):
        for mx_enable in [False, True]:
            bit_width_config = BitWidthConfig()
            quantization_config = QuantizationConfig()
            graph = self.get_test_graph(quantization_config)
            bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])

            updated_graph = set_quantization_configuration_to_graph(
                graph, quantization_config, bit_width_config,
                mixed_precision_enable=mx_enable
            )

            for node in updated_graph.nodes:
                exp_vals = expected.get(node.name)
                if mx_enable == False:
                    assert len(node.candidates_quantization_cfg) == 1
                else:
                    if node.name in expected.keys():
                        assert len(node.candidates_quantization_cfg) == 1
                    else:
                        node_qc_options = node.get_qco(graph.fqc)
                        assert len(node.candidates_quantization_cfg) == len(node_qc_options.quantization_configurations)

                if exp_vals is None: continue

                cfg_list = node.candidates_quantization_cfg[0].weights_quantization_cfg.attributes_config_mapping
                for vkey in cfg_list:
                    cfg = cfg_list.get(vkey)
                    if exp_vals.get(vkey) is not None:
                        assert cfg.weights_n_bits == exp_vals.get(vkey)

    @pytest.mark.parametrize(
        ("inputs", "expected"), [
            (test_input_4, test_expected_4),
            (test_input_5, test_expected_5),
        ])
    def test_manual_weights_bitwidth_selection_error_add(self, inputs, expected):
        for mx_enable in [False, True]:
            bit_width_config = BitWidthConfig()
            quantization_config = QuantizationConfig()
            graph = self.get_test_graph(quantization_config)

            bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])
            try:
                set_quantization_configuration_to_graph(
                    graph, quantization_config, bit_width_config,
                    mixed_precision_enable=mx_enable
                )
            except Exception as e:
                assert expected == str(e)


class TestManualPositionalAttrWeightsBitwidthSelection(TestManualWeightsBitwidthSelection):
    def generate_tpc_local(self, default_config, base_config, mixed_precision_cfg_list):
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
            attr_weights_configs_mapping={POS_ATTR: positional_weight_16_attr_config})

        const_config_input16_positional_weight8 = const_config_input16.clone_and_edit(
            attr_weights_configs_mapping={POS_ATTR: positional_weight_8_attr_config})
        const_configuration_options_inout16 = (
            schema.QuantizationConfigOptions(quantization_configurations=tuple([
                const_config_input16,
                const_config_input16_positional_weight8,
                const_config_input16_positional_weight16]),
                base_config=const_config_input16))

        operator_set = []

        add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD, qc_options=const_configuration_options_inout16)
        operator_set.extend([add])

        generated_tpc = schema.TargetPlatformCapabilities(
            default_qco=default_configuration_options,
            operator_set=tuple(operator_set))

        return generated_tpc

    def get_float_model(self):
        class BaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                const = torch.rand(8)
                self.a = to_torch_tensor(const)

            def forward(self, x):
                x = torch.add(x, self.a)
                return x

        return BaseModel()

    # test case for set_manual_activation_bit_width
    """
    Test Items Policy:
        - How to specify the target layer: Options(type/name)
        - Target attribute information: Options(kernel) 
        - Bit width variations: Options(2, 4, 16)
    """
    test_input_1 = (NodeNameFilter("add"), 16, POS_ATTR)
    test_input_2 = (NodeTypeFilter(torch.add), 8, POS_ATTR)
    test_input_3 = (NodeNameFilter("add"), 4, POS_ATTR)
    test_input_4 = (NodeNameFilter("add"), 2, POS_ATTR)

    test_expected_1 = ({"add": {1: 16}})
    test_expected_2 = ({"add": {1: 8}})
    test_expected_3 = ("Manually selected weights bit-width [[4, 'pos_attr']] is invalid for node "
                       'add:add.')
    test_expected_4 = ("Manually selected weights bit-width [[2, 'pos_attr']] is invalid for node "
                       'add:add.')

    @pytest.mark.parametrize(
        ("inputs", "expected"), [
            (test_input_1, test_expected_1),
            (test_input_2, test_expected_2),
        ])
    def test_manual_weights_bitwidth_selection(self, inputs, expected):
        bit_width_config = BitWidthConfig()
        quantization_config = QuantizationConfig()
        graph = self.get_test_graph(quantization_config)
        bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])

        updated_graph = set_quantization_configuration_to_graph(
            graph, quantization_config, bit_width_config
        )

        for node in updated_graph.nodes:
            exp_vals = expected.get(node.name)
            assert len(node.candidates_quantization_cfg) == 1
            if exp_vals is None:
                continue

            cfg_list = node.candidates_quantization_cfg[0].weights_quantization_cfg.pos_attributes_config_mapping
            for vkey in cfg_list:
                cfg = cfg_list.get(vkey)
                if exp_vals.get(vkey) is not None:
                    assert cfg.weights_n_bits == exp_vals.get(vkey)

    @pytest.mark.parametrize(
        ("inputs", "expected"), [
            (test_input_3, test_expected_3),
            (test_input_4, test_expected_4),
        ])
    def test_manual_weights_bitwidth_selection_error_add(self, inputs, expected):
        super().test_manual_weights_bitwidth_selection_error_add(inputs, expected)
