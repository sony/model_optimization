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
import os

import unittest

import model_compression_toolkit as mct
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import \
    get_config_options_by_operators_set, is_opset_in_model
from tests.common_tests.helpers.generate_test_tp_model import generate_test_attr_configs, generate_test_op_qc

tp = mct.target_platform

TEST_QC = generate_test_op_qc(**generate_test_attr_configs())
TEST_QCO = schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC]))


class TargetPlatformModelingTest(unittest.TestCase):
    def cleanup_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up: {file_path}")

    def test_dump_to_json(self):
        op1 = schema.OperatorsSet(name="opset1")
        op2 = schema.OperatorsSet(name="opset2")
        op3 = schema.OperatorsSet(name="opset3")
        op12 = schema.OperatorSetConcat(operators_set=[op1, op2])
        model = schema.TargetPlatformModel(default_qco=TEST_QCO,
                                           operator_set=(op1, op2, op3),
                                           fusing_patterns=(schema.Fusing(operator_groups=(op12, op3)),
                                                            schema.Fusing(operator_groups=(op1, op2))),
                                           tpc_minor_version=1,
                                           tpc_patch_version=0,
                                           tpc_platform_type="dump_to_json",
                                           add_metadata=False)
        json_str = model.json()
        # Define the output file path
        file_path = "target_platform_model.json"
        # Register cleanup to delete the file if it exists
        self.addCleanup(self.cleanup_file, file_path)

        # Write the JSON string to the file
        with open(file_path, "w") as f:
            f.write(json_str)

        with open(file_path, "r") as f:
            json_content = f.read()

        loaded_target_model = schema.TargetPlatformModel.parse_raw(json_content)
        self.assertEqual(model, loaded_target_model)


    def test_immutable_tp(self):

        with self.assertRaises(Exception) as e:
            model = schema.TargetPlatformModel(default_qco=TEST_QCO,
                                               operator_set=tuple([schema.OperatorsSet(name="opset")]),
                                               tpc_minor_version=None,
                                               tpc_patch_version=None,
                                               tpc_platform_type=None,
                                               add_metadata=False)
            model.operator_set = tuple()
        self.assertEqual('"TargetPlatformModel" is immutable and does not support item assignment', str(e.exception))

    def test_default_options_more_than_single_qc(self):
        test_qco = schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC, TEST_QC]), base_config=TEST_QC)
        with self.assertRaises(Exception) as e:
            schema.TargetPlatformModel(default_qco=test_qco,
                                       tpc_minor_version=None,
                                       tpc_patch_version=None,
                                       tpc_platform_type=None,
                                       add_metadata=False)
        self.assertEqual('Default QuantizationConfigOptions must contain exactly one option.', str(e.exception))

    def test_tp_model_show(self):
        tpm = schema.TargetPlatformModel(default_qco=TEST_QCO,
                                         tpc_minor_version=None,
                                         tpc_patch_version=None,
                                         tpc_platform_type=None,
                                         operator_set=tuple([schema.OperatorsSet(name="opA"), schema.OperatorsSet(name="opB")]),
                                         fusing_patterns=tuple(
                                             [schema.Fusing(operator_groups=(schema.OperatorsSet(name="opA"), schema.OperatorsSet(name="opB")))]),
                                         add_metadata=False)
        tpm.show()

class OpsetTest(unittest.TestCase):

    def test_opset_qco(self):
        opset_name = "ops_3bit"
        qco_3bit = TEST_QCO.clone_and_edit(activation_n_bits=3)
        operator_set = [schema.OperatorsSet(name=opset_name, qc_options=qco_3bit)]
        hm = schema.TargetPlatformModel(default_qco=TEST_QCO,
                                        operator_set=tuple(operator_set),
                                        tpc_minor_version=None,
                                        tpc_patch_version=None,
                                        tpc_platform_type=None,
                                        add_metadata=False,
                                        name='test')
        for op_qc in get_config_options_by_operators_set(hm, opset_name).quantization_configurations:
            self.assertEqual(op_qc.activation_n_bits, 3)

        self.assertTrue(is_opset_in_model(hm, opset_name))
        self.assertFalse(is_opset_in_model(hm, "ShouldNotBeInModel"))
        self.assertEqual(get_config_options_by_operators_set(hm, opset_name), qco_3bit)
        self.assertEqual(get_config_options_by_operators_set(hm, "ShouldNotBeInModel"),
                         hm.default_qco)

    def test_opset_concat(self):
        operator_set, fusing_patterns = [], []

        a = schema.OperatorsSet(name='opset_A')
        b = schema.OperatorsSet(name='opset_B',
                                qc_options=TEST_QCO.clone_and_edit(activation_n_bits=2))
        c = schema.OperatorsSet(name='opset_C')  # Just add it without using it in concat
        operator_set.extend([a, b, c])
        hm = schema.TargetPlatformModel(default_qco=TEST_QCO,
                                        operator_set=tuple(operator_set),
                                        tpc_minor_version=None,
                                        tpc_patch_version=None,
                                        tpc_platform_type=None,
                                        add_metadata=False,
                                        name='test')
        self.assertEqual(len(hm.operator_set), 3)
        self.assertFalse(is_opset_in_model(hm, "opset_A_opset_B"))

    def test_non_unique_opset(self):
        with self.assertRaises(Exception) as e:
            hm = schema.TargetPlatformModel(
                default_qco=schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC])),
                operator_set=tuple([schema.OperatorsSet(name="conv"), schema.OperatorsSet(name="conv")]),
                tpc_minor_version=None,
                tpc_patch_version=None,
                tpc_platform_type=None,
                add_metadata=False)

        self.assertEqual('Operator Sets must have unique names.', str(e.exception))


class QCOptionsTest(unittest.TestCase):

    def test_empty_qc_options(self):
        with self.assertRaises(Exception) as e:
            schema.QuantizationConfigOptions(quantization_configurations=tuple([]))
        self.assertEqual(
            "'QuantizationConfigOptions' requires at least one 'OpQuantizationConfig'. The provided configurations are empty.",
            str(e.exception))

    def test_list_of_no_qc(self):
        with self.assertRaises(Exception) as e:
            schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC, 3]), base_config=TEST_QC)
        self.assertTrue(
            "1 validation error for QuantizationConfigOptions\nquantization_configurations -> 1\n  value is not a valid dict (type=type_error.dict)" in str(
                e.exception))

    def test_clone_and_edit_options(self):
        modified_options = TEST_QCO.clone_and_edit(activation_n_bits=3).clone_and_edit_weight_attribute(
            attrs=[KERNEL_ATTR],
            weights_n_bits=5)

        self.assertEqual(modified_options.quantization_configurations[0].activation_n_bits, 3)
        self.assertEqual(
            modified_options.quantization_configurations[0].attr_weights_configs_mapping[KERNEL_ATTR].weights_n_bits, 5)

    def test_qco_without_base_config(self):
        schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC]))  # Should work fine as it has only one qc.
        with self.assertRaises(Exception) as e:
            schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC, TEST_QC]))  # Should raise exception as base_config was not passed
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
        add = schema.OperatorsSet(name="add")
        with self.assertRaises(Exception) as e:
            hm = schema.TargetPlatformModel(
                default_qco=schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC])),
                operator_set=tuple([add]),
                fusing_patterns=tuple([schema.Fusing(operator_groups=tuple([add]))]),
                tpc_minor_version=None,
                tpc_patch_version=None,
                tpc_platform_type=None,
                add_metadata=False)
        self.assertEqual('Fusing cannot be created for a single operator.', str(e.exception))

    def test_fusing_contains(self):

        operator_set, fusing_patterns = [], []

        conv = schema.OperatorsSet(name="conv")
        add = schema.OperatorsSet(name="add")
        tanh = schema.OperatorsSet(name="tanh")
        operator_set.extend([conv, add, tanh])

        fusing_patterns.append(schema.Fusing(operator_groups=(conv, add)))
        fusing_patterns.append(schema.Fusing(operator_groups=(conv, add, tanh)))

        hm = schema.TargetPlatformModel(
            default_qco=schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC])),
            operator_set=tuple(operator_set),
            fusing_patterns=tuple(fusing_patterns),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False)
        self.assertEqual(len(hm.fusing_patterns), 2)
        f0, f1 = hm.fusing_patterns[0], hm.fusing_patterns[1]
        self.assertTrue(f1.contains(f0))
        self.assertFalse(f0.contains(f1))
        self.assertTrue(f0.contains(f0))
        self.assertTrue(f1.contains(f1))

    def test_fusing_contains_with_opset_concat(self):
        operator_set, fusing_patterns = [], []

        conv = schema.OperatorsSet(name="conv")
        add = schema.OperatorsSet(name="add")
        tanh = schema.OperatorsSet(name="tanh")
        operator_set.extend([conv, add, tanh])

        add_tanh = schema.OperatorSetConcat(operators_set=[add, tanh])
        fusing_patterns.append(schema.Fusing(operator_groups=(conv, add)))
        fusing_patterns.append(schema.Fusing(operator_groups=(conv, add_tanh)))
        fusing_patterns.append(schema.Fusing(operator_groups=(conv, add, tanh)))

        hm = schema.TargetPlatformModel(
            default_qco=schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC])),
            operator_set=tuple(operator_set),
            fusing_patterns=tuple(fusing_patterns),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False)

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
