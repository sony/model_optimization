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
    attach2pytorch = AttachTpcToPytorch()

    tested_op_cfg = default_op_cfg.clone_and_edit(activation_n_bits=42)

    default_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(default_op_cfg,))
    tested_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(tested_op_cfg,))

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
                    assert qco.quantization_configurations[0].activation_n_bits == 42

# def test_attach2pytorch_attach_op_with_attributes():
#     attach2pytorch = AttachTpcToPytorch()
#
#     tested_op_cfg = default_op_cfg.clone_and_edit(activation_n_bits=200)
#
#     default_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(default_op_cfg,))
#     tested_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(tested_op_cfg,))
#
#     for op_name, op_list in attach2pytorch._opset2layer.items():
#         if op_name not in attach2pytorch._opset2attr_mapping.keys():
#             tpc = schema.TargetPlatformModel(
#                 default_qco=default_qc_options,
#                 operator_set=tuple([schema.OperatorsSet(name=op_name, qc_options=tested_qc_options)]))
#
#             pytorch_quant_capabilities = attach2pytorch.attach(tpc)  # Run 'attach' to test operator attach to framework
#
#             assert isinstance(pytorch_quant_capabilities, TargetPlatformCapabilities)
#
#             all_mapped_ops = pytorch_quant_capabilities.layer2qco.copy()
#             all_mapped_ops.update(pytorch_quant_capabilities.filterlayer2qco)
#             if len(op_list) == 0:
#                 assert len(all_mapped_ops) == 0
#             else:
#                 assert len(all_mapped_ops) == len(op_list)
#
#                 for qco in all_mapped_ops.values():
#                     assert len(qco.quantization_configurations) == 1
#                     assert qco.quantization_configurations[0].activation_n_bits == 200


def test_attach2pytorch_attach_to_default_config():
    attach2pytorch = AttachTpcToPytorch()

    default_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(default_op_cfg,))
    opset_name = schema.OperatorSetNames.ADD

    tpc = schema.TargetPlatformModel(default_qco=default_qc_options,
                                     operator_set=tuple([schema.OperatorsSet(name=opset_name)]))

    pytorch_quant_capabilities = attach2pytorch.attach(tpc)  # Run 'attach' to test operator attach to framework

    assert isinstance(pytorch_quant_capabilities, TargetPlatformCapabilities)
    assert pytorch_quant_capabilities.layer2qco.get(opset_name) is not None
    assert pytorch_quant_capabilities.layer2qco[opset_name] == default_qc_options


# TODO:
#  1. Test attach to attribute and attach with default config if not defined in TP model
#  2. Test not existing opset in tp model raise
#  3. Test custom opset name attach + prioritization over builtin
