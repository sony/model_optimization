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
import abc
from typing import NamedTuple

import pytest

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from mct_quantizers import QuantizationMethod
from model_compression_toolkit import DefaultDict
from model_compression_toolkit.core import CustomOpsetLayers
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from model_compression_toolkit.target_platform_capabilities import FrameworkQuantizationCapabilities, \
    LayerFilterParams


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

OpSet = NamedTuple("OpSet", [('op_name', str), ('op_list', list)])


class BaseTpcAttach2FrameworkTest(abc.ABC):

    attach2fw_class = None

    def setup_method(self):
        assert self.attach2fw_class is not None
        self.attach2fw = self.attach2fw_class()


    def test_attach2fw_init(self):
        # verify built-in opset to operator mapping structure
        assert len(self.attach2fw._opset2layer) == 60  # number of built-in operator sets
        assert all(opset in self.attach2fw._opset2layer for opset in list(schema.OperatorSetNames))
        assert all(isinstance(key, schema.OperatorSetNames) for key in self.attach2fw._opset2layer.keys())
        assert all(isinstance(value, list) for value in self.attach2fw._opset2layer.values())

        # verify built-in opset to attributes mapping structure
        assert all(isinstance(key, schema.OperatorSetNames) for key in self.attach2fw._opset2attr_mapping.keys())
        assert all(isinstance(value, dict) for value in self.attach2fw._opset2attr_mapping.values())

    def test_attach2fw_attach_without_attributes(self):
        # Setup TPC with testable configurations
        tested_op_cfg = default_op_cfg.clone_and_edit(activation_n_bits=42)

        default_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(default_op_cfg,))
        tested_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(tested_op_cfg,))

        opsets_without_attrs = [OpSet(op_name=op_name, op_list=op_list)
                                for op_name, op_list in self.attach2fw._opset2layer.items()
                                if op_name not in self.attach2fw._opset2attr_mapping.keys()]

        assert len(opsets_without_attrs) > 0

        for opset in opsets_without_attrs:
            tpc = schema.TargetPlatformCapabilities(
                default_qco=default_qc_options,
                operator_set=tuple([schema.OperatorsSet(name=opset.op_name, qc_options=tested_qc_options)]))

            fw_quant_capabilities = self.attach2fw.attach(tpc)  # Run 'attach' to test operator attach to framework

            assert isinstance(fw_quant_capabilities, FrameworkQuantizationCapabilities)

            all_mapped_ops = fw_quant_capabilities.layer2qco.copy()
            all_mapped_ops.update(fw_quant_capabilities.filterlayer2qco)
            assert len(all_mapped_ops) == len(opset.op_list)

            if len(opset.op_list) > 0:
                for qco in all_mapped_ops.values():
                    assert len(qco.quantization_configurations) == 1
                    assert qco.base_config == tested_op_cfg


    def test_attach2fw_attach_linear_op_with_attributes(self):
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

        opsets_with_attrs = [OpSet(op_name=op_name, op_list=op_list)
                             for op_name, op_list in self.attach2fw._opset2layer.items()
                             if op_name in self.attach2fw._opset2attr_mapping.keys()]

        assert len(opsets_with_attrs) > 0

        for opset in opsets_with_attrs:
            tpc = schema.TargetPlatformCapabilities(
                default_qco=default_qc_options,
                operator_set=tuple([schema.OperatorsSet(name=opset.op_name, qc_options=tested_qc_options)]))

            fw_quant_capabilities = self.attach2fw.attach(tpc)  # Run 'attach' to test operator attach to framework
            fw_linear_attr_names = self.attach2fw._opset2attr_mapping[opset.op_name]

            assert isinstance(fw_quant_capabilities, FrameworkQuantizationCapabilities)

            all_mapped_ops = fw_quant_capabilities.layer2qco.copy()
            all_mapped_ops.update(fw_quant_capabilities.filterlayer2qco)
            assert len(all_mapped_ops) == len(opset.op_list)
            if len(opset.op_list) > 0:

                for qco in all_mapped_ops.values():
                    assert len(qco.quantization_configurations) == 1
                    assert qco.base_config.default_weight_attr_config == default_attr_config

                    for attr_name, fw_layer2attr_mapping in fw_linear_attr_names.items():
                        assert isinstance(fw_layer2attr_mapping, DefaultDict)
                        layer_attr_mapping = fw_layer2attr_mapping.get(opset.op_list[0])
                        assert qco.base_config.attr_weights_configs_mapping[layer_attr_mapping] == tested_attr_cfg


    def test_attach2fw_attach_to_default_config(self):
        default_qc_options = schema.QuantizationConfigOptions(quantization_configurations=(default_op_cfg,))
        opset_name = schema.OperatorSetNames.ADD
        operator_set = schema.OperatorsSet(name=opset_name)

        tpc = schema.TargetPlatformCapabilities(default_qco=default_qc_options,
                                                operator_set=(operator_set,))

        fw_quant_capabilities = self.attach2fw.attach(tpc)

        assert isinstance(fw_quant_capabilities, FrameworkQuantizationCapabilities)
        opset2layer = fw_quant_capabilities.op_sets_to_layers.get_layers_by_op(operator_set)
        assert len(opset2layer) > 0
        opset_cfg = fw_quant_capabilities.layer2qco[opset2layer[0]]
        assert opset_cfg == default_qc_options

    def test_not_existing_opset_with_layers_to_attach(self):
        opset_name = "NotExisting"
        operator_set = schema.OperatorsSet(name=opset_name)

        tpc = schema.TargetPlatformCapabilities(default_qco=schema.QuantizationConfigOptions(
            quantization_configurations=(default_op_cfg,)),
            operator_set=(operator_set,))

        with pytest.raises(Exception, match=f'{opset_name} is defined in TargetPlatformCapabilities'):
            _ = self.attach2fw.attach(tpc)

    def _test_attach2fw_attach_with_custom_opset(self, operators_list, filter_op, fw_attr_name):
        test_bit = 42
        opset_name = "Custom"
        attr_name = "CustomAttr"
        cfg = default_op_cfg.clone_and_edit(attr_weights_configs_mapping=
                                            {attr_name: schema.AttributeQuantizationConfig(weights_n_bits=test_bit)})
        qc_options = schema.QuantizationConfigOptions(quantization_configurations=(cfg,))

        operator_set = schema.OperatorsSet(name=opset_name,
                                           qc_options=qc_options.clone_and_edit(activation_n_bits=test_bit))

        tpc = schema.TargetPlatformCapabilities(default_qco=schema.QuantizationConfigOptions(
            quantization_configurations=(default_op_cfg,)),
            operator_set=(operator_set,))

        with pytest.raises(Exception, match=f'{opset_name} is defined in TargetPlatformCapabilities'):
            _ = self.attach2fw.attach(tpc)

        # Setting a layers mapping for the custom opset with a regular operator and a filter.
        # We also test the option of passing an attributes mapping for the operator to set a specific attribute config.
        fw_custom_attr_name = 'CustomAttrFW'

        assert len(operators_list) == 1
        non_filter_op = operators_list[0]
        layers = [non_filter_op, filter_op]
        fw_quant_capabilities = self.attach2fw.attach(
            tpc,
            custom_opset2layer={opset_name: CustomOpsetLayers(operators=layers,
                                                              attr_mapping={attr_name: DefaultDict(
                                                                  {filter_op: fw_custom_attr_name},
                                                                  default_value=fw_attr_name)})
                                })

        assert isinstance(fw_quant_capabilities, FrameworkQuantizationCapabilities)
        opset_to_layers = fw_quant_capabilities.op_sets_to_layers.op_sets_to_layers
        assert len(opset_to_layers) == 1
        assert opset_to_layers[0].name == opset_name
        assert opset_to_layers[0].layers == layers

        op_cfg = fw_quant_capabilities.layer2qco[non_filter_op].base_config
        assert op_cfg.activation_n_bits == test_bit
        assert fw_attr_name in op_cfg.attr_weights_configs_mapping
        assert op_cfg.attr_weights_configs_mapping[fw_attr_name].weights_n_bits == test_bit

        op_cfg = fw_quant_capabilities.filterlayer2qco[filter_op].base_config
        assert op_cfg.activation_n_bits == test_bit
        assert fw_custom_attr_name in op_cfg.attr_weights_configs_mapping
        assert op_cfg.attr_weights_configs_mapping[fw_custom_attr_name].weights_n_bits == test_bit


    def _test_attach2fw_prioritize_custom_opset(self, op):
        opset_name = schema.OperatorSetNames.CONV

        # setting a custom opset layer mapping to the built-in opset name 'CONV'.
        # we expect the return fw platform capabilities to include the custom operator defined for
        # the opset instead of the built-in list of layers (filter op instead of nn.Conv2d)
        operator_set = schema.OperatorsSet(name=opset_name)

        tpc = schema.TargetPlatformCapabilities(default_qco=schema.QuantizationConfigOptions(
            quantization_configurations=(default_op_cfg,)),
            operator_set=(operator_set,))

        filter_op = LayerFilterParams(op, kernel_size=1)

        fw_quant_capabilities = self.attach2fw.attach(tpc,
                                                      custom_opset2layer={opset_name: CustomOpsetLayers(
                                                          operators=[filter_op])})

        opset_layers = fw_quant_capabilities.op_sets_to_layers.get_layers_by_op(operator_set)
        assert op not in opset_layers
        assert filter_op in opset_layers
