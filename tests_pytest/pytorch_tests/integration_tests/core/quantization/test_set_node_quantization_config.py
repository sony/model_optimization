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
from model_compression_toolkit.core import CoreConfig

from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph

import torch
from torch import nn
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness, \
    AttributeQuantizationConfig
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.constants import FLOAT_BITWIDTH

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation

from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, WEIGHTS_N_BITS
from model_compression_toolkit.target_platform_capabilities.constants import PYTORCH_KERNEL

class TestManualWeightsBitwidthSelection:
    def get_op_qco(self):
        # define a default quantization config for all non-specified weights attributes.
        default_weight_attr_config = AttributeQuantizationConfig(
            weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
            weights_n_bits=8,
            weights_per_channel_threshold=False,
            enable_weights_quantization=False,
            # TODO: this will changed to True once implementing multi-attributes quantization
            lut_values_bitwidth=None)

        # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
        kernel_base_config = AttributeQuantizationConfig(
            weights_quantization_method=QuantizationMethod.SYMMETRIC,
            weights_n_bits=8,
            weights_per_channel_threshold=True,
            enable_weights_quantization=True,
            lut_values_bitwidth=None)

        # define a quantization config to quantize the bias (for layers where there is a bias attribute).
        bias_config = AttributeQuantizationConfig(
            weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
            weights_n_bits=FLOAT_BITWIDTH,
            weights_per_channel_threshold=False,
            enable_weights_quantization=False,
            lut_values_bitwidth=None)

        base_cfg = schema.OpQuantizationConfig(
            default_weight_attr_config=default_weight_attr_config,
            attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config, BIAS_ATTR: bias_config},
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            supported_input_activation_n_bits=8,
            enable_activation_quantization=True,
            quantization_preserving=False,
            fixed_scale=None,
            fixed_zero_point=None,
            simd_size=32,
            signedness=Signedness.AUTO)

        default_config = schema.OpQuantizationConfig(
            default_weight_attr_config=default_weight_attr_config,
            attr_weights_configs_mapping={},
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            supported_input_activation_n_bits=8,
            enable_activation_quantization=True,
            quantization_preserving=False,
            fixed_scale=None,
            fixed_zero_point=None,
            simd_size=32,
            signedness=Signedness.AUTO)

        mx_cfg_list = [base_cfg]
        for n in [2, 4, 16]:
            mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: n}}))
        mx_cfg_list.append(
            base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 4}})
        )

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
            tpc_minor_version=1,
            tpc_patch_version=0,
            tpc_platform_type=None,
            operator_set=tuple(operator_set),
            name='test_manual_weights_bitwidth_selection',
            add_metadata=False,
            is_simd_padding=True)

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

    def get_test_graph(self, core_config):
        float_model = self.get_float_model()
        fw_info = DEFAULT_PYTORCH_INFO

        fw_impl = PytorchImplementation()
        graph = fw_impl.model_reader(float_model,
                                     self.representative_data_gen)
        graph.set_fw_info(fw_info)

        tpc = self.get_tpc()
        attach2pytorch = AttachTpcToPytorch()
        fqc = attach2pytorch.attach(
            tpc, core_config.quantization_config.custom_tpc_opset_to_layer)
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

    test_expected_1 = ({"conv1": {PYTORCH_KERNEL: 2}, "conv2": {PYTORCH_KERNEL: 8}})
    test_expected_2 = ({"conv1": {PYTORCH_KERNEL: 16}, "conv2": {PYTORCH_KERNEL: 16}})
    test_expected_3 = ({"conv1": {PYTORCH_KERNEL: 4}, "conv2": {PYTORCH_KERNEL: 8}})

    @pytest.mark.parametrize(
        ("inputs", "expected"), [
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
        (test_input_3, test_expected_3),
    ])
    def test_manual_weights_bitwidth_selection(self, inputs, expected):
        core_config = CoreConfig()
        graph = self.get_test_graph(core_config)

        core_config.bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])

        updated_graph = set_quantization_configuration_to_graph(
            graph, core_config.quantization_config, core_config.bit_width_config,
            False, False
        )

        for node in updated_graph.nodes:
            exp_vals = expected.get(node.name)
            if exp_vals is None: continue
            assert len(node.candidates_quantization_cfg) == 1

            cfg_list = node.candidates_quantization_cfg[0].weights_quantization_cfg.attributes_config_mapping
            for vkey in cfg_list:
                cfg = cfg_list.get(vkey)
                if exp_vals.get(vkey) is not None:
                    assert cfg.weights_n_bits == exp_vals.get(vkey)

    test_input_4 = (NodeNameFilter("add"), 2, PYTORCH_KERNEL)
    test_expected_4 = ('The requested attribute weight to change the bit width for add:add does not exist.')
    @pytest.mark.parametrize(
        ("inputs", "expected"), [
            (test_input_4, test_expected_4),
    ])
    def test_manual_weights_bitwidth_selection_error_add(self, inputs, expected):
        core_config = CoreConfig()
        graph = self.get_test_graph(core_config)

        core_config.bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])
        try:
            updated_graph = set_quantization_configuration_to_graph(
                graph, core_config.quantization_config, core_config.bit_width_config,
                False, False
            )
        except Exception as e:
            assert expected == str(e)
