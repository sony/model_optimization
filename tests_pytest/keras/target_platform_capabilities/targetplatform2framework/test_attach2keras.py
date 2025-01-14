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
from packaging import version
import tensorflow as tf

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, Identity
else:
    from keras.layers import Conv2D, Identity

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from mct_quantizers import QuantizationMethod
from model_compression_toolkit import DefaultDict
from model_compression_toolkit.core import CustomOpsetLayers
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, KERAS_KERNEL
from model_compression_toolkit.target_platform_capabilities import FrameworkQuantizationCapabilities, \
    LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras


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


def test_attach2keras_init():
    attach2keras = AttachTpcToKeras()

    # verify built-in opset to operator mapping structure
    assert all(isinstance(key, schema.OperatorSetNames) for key in attach2keras._opset2layer.keys()), \
        ("Keys in the operator set-to-layers mapping within the attach-to-framework module must be of the enum "
         "type OperatorSetNames.")

    assert all(isinstance(value, list) for value in attach2keras._opset2layer.values()), \
        ("All values in the operator set-to-layers mapping within the attach-to-framework module must be of "
         "type 'list'.")

    # verify built-in opset to attributes mapping structure
    assert all(isinstance(key, schema.OperatorSetNames) for key in attach2keras._opset2attr_mapping.keys()), \
        ("Keys in the operator set-to-attributes mapping within the attach-to-framework module must be of the enum "
         "type OperatorSetNames.")

    assert all(isinstance(value, dict) for value in attach2keras._opset2attr_mapping.values()), \
        ("All values in the operator set-to-layers mapping within the attach-to-framework module must be of "
         "type 'dict'.")


def test_attach2keras_attach_without_attributes():

    # Setup TPC with testable configurations
    tested_op_cfg = default_op_cfg.clone_and_edit(activation_n_bits=42)

    default_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(default_op_cfg,))
    tested_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(tested_op_cfg,))

    # Test attach to fw for each built-in opset without attributes quantization
    attach2keras = AttachTpcToKeras()

    for op_name, op_list in attach2keras._opset2layer.items():
        if op_name not in attach2keras._opset2attr_mapping.keys():
            tpc = schema.TargetPlatformCapabilities(
                default_qco=default_qc_options,
                operator_set=tuple([schema.OperatorsSet(name=op_name, qc_options=tested_qc_options)]))

            keras_quant_capabilities = attach2keras.attach(tpc)  # Run 'attach' to test operator attach to framework

            assert isinstance(keras_quant_capabilities, FrameworkQuantizationCapabilities)

            all_mapped_ops = keras_quant_capabilities.layer2qco.copy()
            all_mapped_ops.update(keras_quant_capabilities.filterlayer2qco)
            if len(op_list) == 0:
                assert len(all_mapped_ops) == 0
            else:
                assert len(all_mapped_ops) == len(op_list)

                for qco in all_mapped_ops.values():
                    assert len(qco.quantization_configurations) == 1
                    assert qco.base_config.activation_n_bits == 42


def test_attach2keras_attach_linear_op_with_attributes():

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
    attach2keras = AttachTpcToKeras()

    for op_name, op_list in attach2keras._opset2layer.items():
        if op_name in attach2keras._opset2attr_mapping.keys():
            tpc = schema.TargetPlatformCapabilities(
                default_qco=default_qc_options,
                operator_set=tuple([schema.OperatorsSet(name=op_name, qc_options=tested_qc_options)]))

            keras_quant_capabilities = attach2keras.attach(tpc)  # Run 'attach' to test operator attach to framework
            fw_linear_attr_names = attach2keras._opset2attr_mapping[op_name]

            assert isinstance(keras_quant_capabilities, FrameworkQuantizationCapabilities)

            all_mapped_ops = keras_quant_capabilities.layer2qco.copy()
            all_mapped_ops.update(keras_quant_capabilities.filterlayer2qco)
            if len(op_list) == 0:
                assert len(all_mapped_ops) == 0
            else:
                assert len(all_mapped_ops) == len(op_list)

                for qco in all_mapped_ops.values():
                    assert len(qco.quantization_configurations) == 1
                    assert qco.base_config.default_weight_attr_config == default_attr_config

                    for attr_name, fw_layer2attr_mapping in fw_linear_attr_names.items():
                        assert isinstance(fw_layer2attr_mapping, DefaultDict)
                        layer_attr_mapping = fw_layer2attr_mapping.get(op_list[0])
                        assert qco.base_config.attr_weights_configs_mapping.get(layer_attr_mapping) == tested_attr_cfg


def test_attach2keras_attach_to_default_config():
    attach2keras = AttachTpcToKeras()

    default_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(default_op_cfg,))
    opset_name = schema.OperatorSetNames.ADD
    operator_set = schema.OperatorsSet(name=opset_name)

    tpc = schema.TargetPlatformCapabilities(default_qco=default_qc_options,
                                            operator_set=tuple([operator_set]))

    keras_quant_capabilities = attach2keras.attach(tpc)

    assert isinstance(keras_quant_capabilities, FrameworkQuantizationCapabilities)
    opset2layer = keras_quant_capabilities.op_sets_to_layers.get_layers_by_op(operator_set)
    assert len(opset2layer) > 0
    opset_cfg = keras_quant_capabilities.layer2qco.get(opset2layer[0])
    assert opset_cfg is not None
    assert opset_cfg == default_qc_options


def test_attach2keras_attach_with_custom_opset():
    test_bit = 42
    attach2keras = AttachTpcToKeras()
    cfg = default_op_cfg.clone_and_edit(attr_weights_configs_mapping=
                                        {KERNEL_ATTR: schema.AttributeQuantizationConfig(weights_n_bits=test_bit)})
    qc_options = schema.QuantizationConfigOptions(quantization_configurations=(cfg,))
    opset_name = "Custom"
    operator_set = schema.OperatorsSet(name=opset_name,
                                       qc_options=qc_options.clone_and_edit(activation_n_bits=test_bit))

    tpc = schema.TargetPlatformCapabilities(default_qco=schema.QuantizationConfigOptions(
        quantization_configurations=(default_op_cfg,)),
                                     operator_set=tuple([operator_set]))

    with pytest.raises(Exception) as e_info:
        _ = attach2keras.attach(tpc)
    assert f'{opset_name} is defined in TargetPlatformCapabilities' in str(e_info)

    # Setting a layers mapping for the custom opset with a regular operator and a filter.
    # We also test the option of passing an attributes mapping for the operator to set a specific attribute config.
    custom_attr_name = 'CustomAttr'
    filter_op = LayerFilterParams(Conv2D, stride=2)
    keras_quant_capabilities = attach2keras.attach(
        tpc,
        custom_opset2layer={opset_name: CustomOpsetLayers(operators=[Identity,
                                                                     filter_op],
                                                          attr_mapping={KERNEL_ATTR: DefaultDict(
                                                              {filter_op: custom_attr_name},
                                                              default_value=KERAS_KERNEL)})
                            })

    assert isinstance(keras_quant_capabilities, FrameworkQuantizationCapabilities)
    opset_to_layers = keras_quant_capabilities.op_sets_to_layers.op_sets_to_layers
    assert len(opset_to_layers) == 1
    assert opset_to_layers[0].name == opset_name
    assert len(opset_to_layers[0].layers) == 2

    op_cfg = keras_quant_capabilities.layer2qco[Identity].base_config
    assert op_cfg.activation_n_bits == test_bit
    assert KERAS_KERNEL in op_cfg.attr_weights_configs_mapping
    assert op_cfg.attr_weights_configs_mapping[KERAS_KERNEL].weights_n_bits == test_bit

    op_cfg = keras_quant_capabilities.filterlayer2qco[filter_op].base_config
    assert op_cfg.activation_n_bits == test_bit
    assert custom_attr_name in op_cfg.attr_weights_configs_mapping
    assert op_cfg.attr_weights_configs_mapping[custom_attr_name].weights_n_bits == test_bit


def test_attach2keras_prioritize_custom_opset():
    attach2keras = AttachTpcToKeras()
    opset_name = schema.OperatorSetNames.CONV

    # setting a custom opset layer mapping to the built-in opset name 'CONV'.
    # we expect the return keras platform capabilities to include the custom operator defined for
    # the opset instead of the built-in list of layers (filter op instead of nn.Conv2d)
    operator_set = schema.OperatorsSet(name=opset_name)

    tpc = schema.TargetPlatformCapabilities(default_qco=schema.QuantizationConfigOptions(
        quantization_configurations=(default_op_cfg,)),
        operator_set=tuple([operator_set]))

    filter_op = LayerFilterParams(Conv2D, kernel_size=1)
    keras_quant_capabilities = attach2keras.attach(tpc,
                                                   custom_opset2layer={opset_name:
                                                                           CustomOpsetLayers(operators=[filter_op])})

    opset_layers = keras_quant_capabilities.op_sets_to_layers.get_layers_by_op(operator_set)
    assert Conv2D not in opset_layers
    assert filter_op in opset_layers


def test_not_existing_opset_with_layers_to_attach():
    attach2keras = AttachTpcToKeras()
    opset_name = "NotExisting"
    operator_set = schema.OperatorsSet(name=opset_name)

    tpc = schema.TargetPlatformCapabilities(default_qco=schema.QuantizationConfigOptions(
        quantization_configurations=(default_op_cfg,)),
        operator_set=tuple([operator_set]))

    with pytest.raises(Exception) as e_info:
        _ = attach2keras.attach(tpc)
    assert f'{opset_name} is defined in TargetPlatformCapabilities' in str(e_info)
