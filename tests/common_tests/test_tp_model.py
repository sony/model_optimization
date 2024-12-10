# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

import unittest

import model_compression_toolkit as mct
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from model_compression_toolkit.target_platform_capabilities.target_platform import \
    get_default_quantization_config_options
from tests.common_tests.helpers.generate_test_tp_model import generate_test_attr_configs, generate_test_op_qc

tp = mct.target_platform

TEST_QC = generate_test_op_qc(**generate_test_attr_configs())
TEST_QCO = schema.QuantizationConfigOptions([TEST_QC])


class TargetPlatformModelingTest(unittest.TestCase):

    def test_not_initialized_tp(self):
        with self.assertRaises(Exception) as e:
            mct.target_platform.get_default_quantization_config_options()
        self.assertEqual('Target platform model is not initialized.', str(e.exception))

    def test_get_default_options(self):
        with schema.TargetPlatformModel(TEST_QCO,
                                        tpc_minor_version=None,
                                        tpc_patch_version=None,
                                        tpc_platform_type=None,
                                        add_metadata=False):
            self.assertEqual(tp.get_default_quantization_config_options(), TEST_QCO)

    def test_immutable_tp(self):
        model = schema.TargetPlatformModel(TEST_QCO,
                                           tpc_minor_version=None,
                                           tpc_patch_version=None,
                                           tpc_platform_type=None,
                                           add_metadata=False)
        with self.assertRaises(Exception) as e:
            with model:
                schema.OperatorsSet("opset")
            model.operator_set = []
        self.assertEqual("cannot assign to field 'operator_set'", str(e.exception))

    def test_default_options_more_than_single_qc(self):
        test_qco = schema.QuantizationConfigOptions([TEST_QC, TEST_QC], base_config=TEST_QC)
        with self.assertRaises(Exception) as e:
            schema.TargetPlatformModel(test_qco,
                                       tpc_minor_version=None,
                                       tpc_patch_version=None,
                                       tpc_platform_type=None,
                                       add_metadata=False)
        self.assertEqual('Default QuantizationConfigOptions must contain exactly one option.', str(e.exception))

    def test_tp_model_show(self):
        tpm = schema.TargetPlatformModel(TEST_QCO,
                                         tpc_minor_version=None,
                                         tpc_patch_version=None,
                                         tpc_platform_type=None,
                                         add_metadata=False)
        with tpm:
            a = schema.OperatorsSet("opA")


class OpsetTest(unittest.TestCase):

    def test_opset_qco(self):
        hm = schema.TargetPlatformModel(TEST_QCO,
                                        tpc_minor_version=None,
                                        tpc_patch_version=None,
                                        tpc_platform_type=None,
                                        add_metadata=False,
                                        name='test')
        opset_name = "ops_3bit"
        with hm:
            qco_3bit = get_default_quantization_config_options().clone_and_edit(activation_n_bits=3)
            schema.OperatorsSet(opset_name, qco_3bit)

        for op_qc in hm.get_config_options_by_operators_set(opset_name).quantization_config_list:
            self.assertEqual(op_qc.activation_n_bits, 3)

        self.assertTrue(hm.is_opset_in_model(opset_name))
        self.assertFalse(hm.is_opset_in_model("ShouldNotBeInModel"))
        self.assertEqual(hm.get_config_options_by_operators_set(opset_name), qco_3bit)
        self.assertEqual(hm.get_config_options_by_operators_set("ShouldNotBeInModel"),
                         hm.default_qco)

    def test_opset_concat(self):
        hm = schema.TargetPlatformModel(TEST_QCO,
                                        tpc_minor_version=None,
                                        tpc_patch_version=None,
                                        tpc_platform_type=None,
                                        add_metadata=False,
                                        name='test')
        with hm:
            a = schema.OperatorsSet('opset_A')
            b = schema.OperatorsSet('opset_B',
                                    get_default_quantization_config_options().clone_and_edit(activation_n_bits=2))
            schema.OperatorsSet('opset_C')  # Just add it without using it in concat
            schema.OperatorSetConcat([a, b])
        self.assertEqual(len(hm.operator_set), 4)
        self.assertTrue(hm.is_opset_in_model("opset_A_opset_B"))
        self.assertTrue(hm.get_config_options_by_operators_set('opset_A_opset_B') is None)

    def test_non_unique_opset(self):
        hm = schema.TargetPlatformModel(
            schema.QuantizationConfigOptions([TEST_QC]),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False)
        with self.assertRaises(Exception) as e:
            with hm:
                schema.OperatorsSet("conv")
                schema.OperatorsSet("conv")
        self.assertEqual('Operator Sets must have unique names.', str(e.exception))


class QCOptionsTest(unittest.TestCase):

    def test_empty_qc_options(self):
        with self.assertRaises(Exception) as e:
            schema.QuantizationConfigOptions([])
        self.assertEqual(
            "'QuantizationConfigOptions' requires at least one 'OpQuantizationConfig'. The provided list is empty.",
            str(e.exception))

    def test_list_of_no_qc(self):
        with self.assertRaises(Exception) as e:
            schema.QuantizationConfigOptions([TEST_QC, 3])
        self.assertEqual(
            'Each option must be an instance of \'OpQuantizationConfig\', but found an object of type: <class \'int\'>.',
            str(e.exception))

    def test_clone_and_edit_options(self):
        modified_options = TEST_QCO.clone_and_edit(activation_n_bits=3).clone_and_edit_weight_attribute(
            attrs=[KERNEL_ATTR],
            weights_n_bits=5)

        self.assertEqual(modified_options.quantization_config_list[0].activation_n_bits, 3)
        self.assertEqual(
            modified_options.quantization_config_list[0].attr_weights_configs_mapping[KERNEL_ATTR].weights_n_bits, 5)

    def test_qco_without_base_config(self):
        schema.QuantizationConfigOptions([TEST_QC])  # Should work fine as it has only one qc.
        with self.assertRaises(Exception) as e:
            schema.QuantizationConfigOptions([TEST_QC, TEST_QC])  # Should raise exception as base_config was not passed
        self.assertEqual(
            'For multiple configurations, a \'base_config\' is required for non-mixed-precision optimization.',
            str(e.exception))

    def test_get_qco_for_none_tpc(self):
        mock_node = BaseNode(name="", framework_attr={}, input_shape=(), output_shape=(), weights={}, layer_class=None)
        with self.assertRaises(Exception) as e:
            mock_node.get_qco(None)
        self.assertEqual('Can not retrieve QC options for None TPC', str(e.exception))


class FusingTest(unittest.TestCase):

    def test_fusing_single_opset(self):
        hm = schema.TargetPlatformModel(
            schema.QuantizationConfigOptions([TEST_QC]),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False)
        with hm:
            add = schema.OperatorsSet("add")
            with self.assertRaises(Exception) as e:
                schema.Fusing([add])
            self.assertEqual('Fusing cannot be created for a single operator.', str(e.exception))

    def test_fusing_contains(self):
        hm = schema.TargetPlatformModel(
            schema.QuantizationConfigOptions([TEST_QC]),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False)
        with hm:
            conv = schema.OperatorsSet("conv")
            add = schema.OperatorsSet("add")
            tanh = schema.OperatorsSet("tanh")
            schema.Fusing([conv, add])
            schema.Fusing([conv, add, tanh])

        self.assertEqual(len(hm.fusing_patterns), 2)
        f0, f1 = hm.fusing_patterns[0], hm.fusing_patterns[1]
        self.assertTrue(f1.contains(f0))
        self.assertFalse(f0.contains(f1))
        self.assertTrue(f0.contains(f0))
        self.assertTrue(f1.contains(f1))

    def test_fusing_contains_with_opset_concat(self):
        hm = schema.TargetPlatformModel(
            schema.QuantizationConfigOptions([TEST_QC]),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False)
        with hm:
            conv = schema.OperatorsSet("conv")
            add = schema.OperatorsSet("add")
            tanh = schema.OperatorsSet("tanh")
            add_tanh = schema.OperatorSetConcat([add, tanh])
            schema.Fusing([conv, add])
            schema.Fusing([conv, add_tanh])
            schema.Fusing([conv, add, tanh])

        self.assertEqual(len(hm.fusing_patterns), 3)
        f0, f1, f2 = hm.fusing_patterns[0], hm.fusing_patterns[1], hm.fusing_patterns[2]

        self.assertTrue(f0.contains(f0))
        self.assertTrue(f1.contains(f1))
        self.assertTrue(f2.contains(f2))

        self.assertTrue(f2.contains(f0))
        self.assertTrue(f1.contains(f0))

        self.assertFalse(f0.contains(f1))
        self.assertFalse(f0.contains(f2))

        self.assertFalse(f2.contains(f1))
        self.assertFalse(f1.contains(f2))
