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
from unittest.mock import MagicMock, PropertyMock, Mock

import pytest

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from mct_quantizers import QuantizationMethod
from model_compression_toolkit import DefaultDict
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch


default_op_cfg = schema.OpQuantizationConfig(
        default_weight_attr_config=schema.AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=[],
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=schema.Signedness.AUTO
    )


def test_attach2pytorch_init():
    attach2pytorch = AttachTpcToPytorch()

    # verify built-in opset to operator mapping structure
    assert all(isinstance(key, schema.OperatorSetNames) for key in attach2pytorch._opset2layer.keys()), \
        ("Keys in the operator set-to-layers mapping within the attach-to-framework module must be of the enum "
         "type OperatorSetNames.")

    assert all(isinstance(value, list) for value in attach2pytorch._opset2layer.values()), \
        ("All values in the operator set-to-layers mapping within the attach-to-framework module must be of "
         "type 'list'.")

    # verify built-in opset to attributes mapping structure
    assert all(isinstance(key, schema.OperatorSetNames) for key in attach2pytorch._opset2attr_mapping.keys()), \
        ("Keys in the operator set-to-attributes mapping within the attach-to-framework module must be of the enum "
         "type OperatorSetNames.")

    assert all(isinstance(value, dict) for value in attach2pytorch._opset2attr_mapping.values()), \
        ("All values in the operator set-to-layers mapping within the attach-to-framework module must be of "
         "type 'dict'.")


def test_attach2pytorch_attach_without_attributes():

    # Setup TPC with testable configurations
    tested_op_cfg = default_op_cfg.clone_and_edit(activation_n_bits=42)

    default_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(default_op_cfg,))
    tested_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(tested_op_cfg,))

    # Test attach to fw for each built-in opset without attributes quantization
    attach2pytorch = AttachTpcToPytorch()

    for op_name, op_list in attach2pytorch._opset2layer.items():
        if op_name not in attach2pytorch._opset2attr_mapping.keys():
            tpc = schema.TargetPlatformModel(
                default_qco=default_qc_options,
                operator_set=tuple([schema.OperatorsSet(name=op_name, qc_options=tested_qc_options)]))

            pytorch_quant_capabilities = attach2pytorch.attach(tpc)  # Run 'attach' to test operator attach to framework

            assert isinstance(pytorch_quant_capabilities, TargetPlatformCapabilities)

            all_mapped_ops = pytorch_quant_capabilities.layer2qco.copy()
            all_mapped_ops.update(pytorch_quant_capabilities.filterlayer2qco)
            if len(op_list) == 0:
                assert len(all_mapped_ops) == 0
            else:
                assert len(all_mapped_ops) == len(op_list)

                for qco in all_mapped_ops.values():
                    assert len(qco.quantization_configurations) == 1
                    assert qco.base_config.activation_n_bits == 42


def test_attach2pytorch_attach_linear_op_with_attributes():

    # Setup TPC with testable configurations
    default_attr_config = schema.AttributeQuantizationConfig(
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_n_bits=8,
        weights_per_channel_threshold=False,
        enable_weights_quantization=False,
        lut_values_bitwidth=None)

    tested_attr_cfg = default_attr_config.clone_and_edit(weights_n_bits=42)

    tested_op_cfg = default_op_cfg.clone_and_edit(default_weight_attr_config=default_attr_config,
                                                  attr_weights_configs_mapping={KERNEL_ATTR: tested_attr_cfg,
                                                                                BIAS_ATTR: tested_attr_cfg})

    default_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(default_op_cfg,))
    tested_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(tested_op_cfg,))

    # Test attach to fw for each built-in opset with attributes quantization
    attach2pytorch = AttachTpcToPytorch()

    for op_name, op_list in attach2pytorch._opset2layer.items():
        if op_name in attach2pytorch._opset2attr_mapping.keys():
            tpc = schema.TargetPlatformModel(
                default_qco=default_qc_options,
                operator_set=tuple([schema.OperatorsSet(name=op_name, qc_options=tested_qc_options)]))

            pytorch_quant_capabilities = attach2pytorch.attach(tpc)  # Run 'attach' to test operator attach to framework
            fw_linear_attr_names = attach2pytorch._opset2attr_mapping[op_name]

            assert isinstance(pytorch_quant_capabilities, TargetPlatformCapabilities)

            all_mapped_ops = pytorch_quant_capabilities.layer2qco.copy()
            all_mapped_ops.update(pytorch_quant_capabilities.filterlayer2qco)
            if len(op_list) == 0:
                assert len(all_mapped_ops) == 0
            else:
                assert len(all_mapped_ops) == len(op_list)

                for qco in all_mapped_ops.values():
                    assert len(qco.quantization_configurations) == 1
                    assert qco.base_config.default_weight_attr_config == default_attr_config

                    for attr_name, fw_layer2attr_mapping in fw_linear_attr_names.items():
                        assert isinstance(fw_layer2attr_mapping, DefaultDict)
                        layer_attr_mapping = fw_layer2attr_mapping.get(attr_name)
                        assert qco.base_config.attr_weights_configs_mapping.get(layer_attr_mapping) == tested_attr_cfg


def test_attach2pytorch_attach_to_default_config():
    attach2pytorch = AttachTpcToPytorch()

    default_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(default_op_cfg,))
    opset_name = schema.OperatorSetNames.ADD
    operator_set = schema.OperatorsSet(name=opset_name)

    tpc = schema.TargetPlatformModel(default_qco=default_qc_options,
                                     operator_set=tuple([operator_set]))

    pytorch_quant_capabilities = attach2pytorch.attach(tpc)  # Run 'attach' to test operator attach to framework

    assert isinstance(pytorch_quant_capabilities, TargetPlatformCapabilities)
    opset2layer = pytorch_quant_capabilities.op_sets_to_layers.get_layers_by_op(operator_set)
    assert len(opset2layer) > 0
    opset_cfg = pytorch_quant_capabilities.layer2qco.get(opset2layer[0])
    assert opset_cfg is not None
    assert opset_cfg == default_qc_options


# TODO:
#  1. Test not existing opset in tp model raise
#  2. Test custom opset name attach + prioritization over builtin
