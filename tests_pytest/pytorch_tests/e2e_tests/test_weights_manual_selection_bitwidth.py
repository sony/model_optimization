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
from tests.common_tests.helpers.tpcs_for_tests.v2.tpc import get_tpc
from model_compression_toolkit.target_platform_capabilities.constants import BIAS, PYTORCH_KERNEL
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, WEIGHTS_N_BITS
from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs, generate_tpc
from mct_quantizers import PytorchQuantizationWrapper

kernel_weights_n_bits = 8
bias_weights_n_bits = 32
activation_n_bits = 8

def representative_data_gen(shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
    for _ in range(num_iter):
        yield [torch.randn(batch_size, *shape)] * num_inputs

def get_tpc():
    base_cfg, _, default_config = get_op_quantization_configs()
    base_cfg = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                        {
                                            KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                        .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                            BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                        .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                        },
                                        activation_n_bits=activation_n_bits)

    mx_cfg_list = [base_cfg]
    for n in [2, 4, 16]:
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: n}}))
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={BIAS_ATTR: {WEIGHTS_N_BITS: n}}))
    mx_cfg_list.append(
        base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 4}, BIAS_ATTR: {WEIGHTS_N_BITS: 16}})
    )
    tpc = generate_tpc(default_config=default_config, base_config=base_cfg, mixed_precision_cfg_list=mx_cfg_list, name='imx500_tpc_test')
    return tpc

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


class TestManualWeightsBitwidthSelectionByLayerType:

    test_input_1 = (NodeTypeFilter(Conv2d), 16, PYTORCH_KERNEL)
    test_input_2 = (NodeTypeFilter(Conv2d), [2], [PYTORCH_KERNEL])
    
    test_expected_1 = ([Conv2d], [16])
    test_expected_2 = ([Conv2d], [2])
    
    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
    ])

    def test_manual_weights_bitwidth_selection(self, inputs, expected):

        float_model = get_float_model()

        target_platform_cap = get_tpc()
        
        core_config = CoreConfig()
        core_config.bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])
        
        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=float_model,
            representative_data_gen=representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=target_platform_cap
        )

        for name, layer in quantized_model.named_modules():
            # check if the layer is a weights quantizer
            if isinstance(layer, PytorchQuantizationWrapper):
                expected_bitwidths = expected[1]
                attrs = inputs[2]

                if not isinstance(attrs, list):
                    attrs = [attrs]

                for bitwidth, attr in zip(expected_bitwidths, attrs):
                    
                    if layer.weights_quantizers.get(attr) is not None:
                        assert layer.weights_quantizers.get(attr).num_bits == bitwidth


class TestManualWeightsBitwidthSelectionByLayerName:

    test_input_1 = (NodeNameFilter("conv1"), 16, PYTORCH_KERNEL)
    test_input_2 = (NodeNameFilter("conv1"), [2], [PYTORCH_KERNEL])
    test_input_3 = ([NodeNameFilter("conv1"), NodeNameFilter("conv1")], [4, 16], [PYTORCH_KERNEL, BIAS])

    test_expected_1 = (["conv1"], [16])
    test_expected_2 = (["conv1"], [2])
    test_expected_3 = (["conv1", "conv1"], [4, 16])
    
    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
        (test_input_3, test_expected_3),
    ])

    def test_manual_weights_bitwidth_selection(self, inputs, expected):

        float_model = get_float_model()

        target_platform_cap = get_tpc()
        
        core_config = CoreConfig()
        core_config.bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])
        
        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=float_model,
            representative_data_gen=representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=target_platform_cap
        )

        for name, layer in quantized_model.named_modules():
            # check if the layer is a weights quantizer
            if isinstance(layer, PytorchQuantizationWrapper):
                bitwidths = expected[1]
                attrs = inputs[2]

                if name == "conv1":
                    for attr, bitwidth in zip(attrs, bitwidths):
                        if layer.weights_quantizers.get(attr) is not None:
                            assert layer.weights_quantizers.get(attr).num_bits == bitwidth
                else:
                    for attr in attrs:
                        if layer.weights_quantizers.get(attr) is not None:
                                if attr == PYTORCH_KERNEL:
                                    assert layer.weights_quantizers.get(attr).num_bits == kernel_weights_n_bits
                                elif attr == BIAS:
                                    assert layer.weights_quantizers.get(attr).num_bits == bias_weights_n_bits
