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
from typing import List

import model_compression_toolkit as mct
import torch
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchPreservingActivationQuantizationHolder

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, IMX500_TP_MODEL
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities, OpQuantizationConfig

def representative_data_gen(shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
    for _ in range(num_iter):
        yield [torch.randn(batch_size, *shape)] * num_inputs

def get_float_model():
        class BaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.dropout = torch.nn.Dropout()
                self.flatten = torch.nn.Flatten()
                self.fc = torch.nn.Linear(48, 128)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.dropout(x)
                x = self.flatten(x)
                x = self.fc(x)
                return x
        return BaseModel()

def generate_tpc(default_config: OpQuantizationConfig,
                 base_config: OpQuantizationConfig,
                 mixed_precision_cfg_list: List[OpQuantizationConfig],
                 name: str) -> TargetPlatformCapabilities:
    
    default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([default_config]))
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                             base_config=base_config)

    operator_set = []
    fusing_patterns = []

    preserving_quantization_config = (default_configuration_options.clone_and_edit(enable_activation_quantization=False, quantization_preserving=True)
                              .clone_and_edit_weight_attribute(enable_weights_quantization=False))

    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.DROPOUT, qc_options=preserving_quantization_config))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.FLATTEN, qc_options=preserving_quantization_config))

    conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
    conv_transpose = schema.OperatorsSet(name=schema.OperatorSetNames.CONV_TRANSPOSE, qc_options=mixed_precision_configuration_options)
    depthwise_conv = schema.OperatorsSet(name=schema.OperatorSetNames.DEPTHWISE_CONV, qc_options=mixed_precision_configuration_options)
    fc = schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED, qc_options=mixed_precision_configuration_options)

    operator_set.extend([conv, conv_transpose, depthwise_conv, fc])

    generated_tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        tpc_minor_version=1,
        tpc_patch_version=0,
        tpc_platform_type=IMX500_TP_MODEL,
        operator_set=tuple(operator_set),
        fusing_patterns=tuple(fusing_patterns),
        name=name,
        add_metadata=False,
        is_simd_padding=True)
    return generated_tpc

def get_tpc():
    base_cfg, mx_cfg_list, default_config = get_op_quantization_configs()

    base_cfg = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                {
                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                .clone_and_edit(enable_weights_quantization=False),
                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                .clone_and_edit(enable_weights_quantization=False),
                                },
                            )

    for i, mx_cfg in enumerate(mx_cfg_list):
        mx_cfg_list[i] = mx_cfg.clone_and_edit(attr_weights_configs_mapping=
                                    {
                                        KERNEL_ATTR: mx_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                    .clone_and_edit(enable_weights_quantization=False),
                                        BIAS_ATTR: mx_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                    .clone_and_edit(enable_weights_quantization=False),
                                    },
                                )
    tpc = generate_tpc(default_config=default_config, base_config=base_cfg, mixed_precision_cfg_list=mx_cfg_list, name='imx500_tpc_test')
    return tpc

test_input_0 = ("conv1_activation_holder_quantizer",)
test_input_1 = ("conv2_activation_holder_quantizer",)
test_input_2 = ("dropout_activation_holder_quantizer",)
test_input_3 = ("flatten_activation_holder_quantizer",)
test_input_4 = ("fc_activation_holder_quantizer",)

test_expected_0 = (PytorchActivationQuantizationHolder,)
test_expected_1 = (PytorchActivationQuantizationHolder,)
test_expected_2 = (PytorchPreservingActivationQuantizationHolder,)
test_expected_3 = (PytorchPreservingActivationQuantizationHolder,)
test_expected_4 = (PytorchActivationQuantizationHolder,)

@pytest.mark.parametrize(("inputs", "expected"), [
    (test_input_0, test_expected_0),
    (test_input_1, test_expected_1),
    (test_input_2, test_expected_2),
    (test_input_3, test_expected_3),
    (test_input_4, test_expected_4),
])
def test_quantization_preserving_holder(inputs, expected):

    float_model = get_float_model()
    target_platform_cap = get_tpc()
    core_config = CoreConfig()
    
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model,
        representative_data_gen=representative_data_gen,
        core_config=core_config,
        target_platform_capabilities=target_platform_cap
    )

    preserving_activation_holder_quantizer_name = ["dropout_activation_holder_quantizer", "flatten_activation_holder_quantizer"]
    activation_holder_quantizer_name = ["conv1_activation_holder_quantizer", "conv2_activation_holder_quantizer", "fc_activation_holder_quantizer"]

    for name, layer in quantized_model.named_modules():
        if name == inputs[0]:
            if name in preserving_activation_holder_quantizer_name:
                assert isinstance(layer, expected[0]) # check holder
                assert layer.quantization_bypass == True

            elif name in activation_holder_quantizer_name:
                assert isinstance(layer, expected[0]) # check holder
             