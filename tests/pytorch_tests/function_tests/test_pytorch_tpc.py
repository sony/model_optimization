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
import copy
import unittest
from functools import partial
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import Hardtanh
from torch.nn.functional import hardtanh
from torchvision.models import mobilenet_v2

import model_compression_toolkit as mct
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.core import MixedPrecisionQuantizationConfig
from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.constants import PYTORCH, FUSED_LAYER_PATTERN
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.quantization_preparation.load_fqc import fetch_qc_options_for_node
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL, IMX500_TP_MODEL, \
    TFLITE_TP_MODEL, QNNPACK_TP_MODEL, KERNEL_ATTR, WEIGHTS_N_BITS, PYTORCH_KERNEL, BIAS_ATTR, BIAS
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework import LayerFilterParams, \
    OperationsSetToLayers
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attribute_filter import Greater, \
    Smaller, Eq
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities
from tests.common_tests.helpers.generate_test_tpc import generate_test_op_qc, generate_test_attr_configs
from tests.pytorch_tests.layer_tests.base_pytorch_layer_test import LayerTestModel


TEST_QC = generate_test_op_qc(**generate_test_attr_configs())
TEST_QCO = schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC]))


class TestPytorchTPModel(unittest.TestCase):

    def test_pytorch_layers_with_params(self):
        hardtanh_with_params = LayerFilterParams(Hardtanh, Greater("max_val", 2))
        self.assertTrue(get_node(Hardtanh(max_val=3)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(get_node(Hardtanh(max_val=2)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(get_node(Hardtanh(max_val=1)).is_match_filter_params(hardtanh_with_params))

        hardtanh_with_params = LayerFilterParams(hardtanh, Greater("max_val", 2))
        self.assertTrue(get_node(partial(hardtanh, max_val=3)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(get_node(partial(hardtanh, max_val=2)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(get_node(partial(hardtanh, max_val=1)).is_match_filter_params(hardtanh_with_params))

        hardtanh_with_params = LayerFilterParams(Hardtanh, Greater("max_val", 2) & Smaller("min_val", 1))
        self.assertTrue(get_node(Hardtanh(max_val=3, min_val=0)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(get_node(Hardtanh(max_val=3, min_val=1)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(get_node(Hardtanh(max_val=2, min_val=0.5)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(get_node(Hardtanh(max_val=2)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(get_node(Hardtanh(max_val=1, min_val=0.5)).is_match_filter_params(hardtanh_with_params))

        hardtanh_with_params = LayerFilterParams(hardtanh, Greater("max_val", 2) & Smaller("min_val", 1))
        self.assertTrue(get_node(partial(hardtanh, max_val=3, min_val=0)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(get_node(partial(hardtanh, max_val=3, min_val=1)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(
            get_node(partial(hardtanh, max_val=2, min_val=0.5)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(get_node(partial(hardtanh, max_val=2)).is_match_filter_params(hardtanh_with_params))
        self.assertFalse(
            get_node(partial(hardtanh, max_val=1, min_val=0.5)).is_match_filter_params(hardtanh_with_params))

        l2norm_tflite_opset = LayerFilterParams(torch.nn.functional.normalize, Eq('p', 2) | Eq('p', None))
        self.assertTrue(
            get_node(partial(torch.nn.functional.normalize, p=2)).is_match_filter_params(l2norm_tflite_opset))
        self.assertTrue(
            get_node(partial(torch.nn.functional.normalize, p=2.0)).is_match_filter_params(l2norm_tflite_opset))
        self.assertTrue(get_node(torch.nn.functional.normalize).is_match_filter_params(l2norm_tflite_opset))
        self.assertFalse(
            get_node(partial(torch.nn.functional.normalize, p=3.0)).is_match_filter_params(l2norm_tflite_opset))

    def test_qco_by_pytorch_layer(self):
        default_qco = schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC]))
        default_qco = default_qco.clone_and_edit(attr_weights_configs_mapping={})
        mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(
            [TEST_QC,
             TEST_QC.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 4}}),
             TEST_QC.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 2}})]),
            base_config=TEST_QC)

        operator_set = []
        operator_set.append(schema.OperatorsSet(name="conv", qc_options=mixed_precision_configuration_options))

        sevenbit_qco = TEST_QCO.clone_and_edit(activation_n_bits=7,
                                               attr_weights_configs_mapping={})
        operator_set.append(schema.OperatorsSet(name="tanh", qc_options=sevenbit_qco))

        sixbit_qco = TEST_QCO.clone_and_edit(activation_n_bits=6,
                                             attr_weights_configs_mapping={})
        operator_set.append(schema.OperatorsSet(name="avg_pool2d_kernel_2", qc_options=sixbit_qco))

        operator_set.append(schema.OperatorsSet(name="avg_pool2d"))

        tpm = schema.TargetPlatformCapabilities(default_qco=default_qco,
                                                tpc_minor_version=None,
                                                tpc_patch_version=None,
                                                tpc_platform_type=None,
                                                operator_set=tuple(operator_set),
                                                add_metadata=False,
                                                name='test')

        tpc_pytorch = FrameworkQuantizationCapabilities(tpm)
        with tpc_pytorch:
            OperationsSetToLayers("conv", [torch.nn.Conv2d],
                                     attr_mapping={KERNEL_ATTR: DefaultDict(default_value=PYTORCH_KERNEL),
                                                   BIAS_ATTR: DefaultDict(default_value=BIAS)})
            OperationsSetToLayers("tanh", [torch.tanh])
            OperationsSetToLayers("avg_pool2d_kernel_2",
                                     [LayerFilterParams(torch.nn.functional.avg_pool2d, kernel_size=2)])
            OperationsSetToLayers("avg_pool2d",
                                     [torch.nn.functional.avg_pool2d])

        with patch('model_compression_toolkit.core.common.framework_info._current_framework_info'):
            conv_node = get_node(torch.nn.Conv2d(3, 3, (1, 1)))
            tanh_node = get_node(torch.tanh)
            avg_pool2d_k2 = get_node(partial(torch.nn.functional.avg_pool2d, kernel_size=2))
            avg_pool2d = get_node(partial(torch.nn.functional.avg_pool2d, kernel_size=1))

        conv_qco = fetch_qc_options_for_node(conv_node, tpc_pytorch)
        tanh_qco = fetch_qc_options_for_node(tanh_node, tpc_pytorch)
        avg_pool2d_k2_qco = fetch_qc_options_for_node(avg_pool2d_k2, tpc_pytorch)
        avg_pool2d_qco = fetch_qc_options_for_node(avg_pool2d, tpc_pytorch)

        self.assertEqual(len(conv_qco.quantization_configurations),
                         len(mixed_precision_configuration_options.quantization_configurations))
        for i in range(len(conv_qco.quantization_configurations)):
            self.assertEqual(conv_qco.quantization_configurations[i].attr_weights_configs_mapping[PYTORCH_KERNEL],
                             mixed_precision_configuration_options.quantization_configurations[
                                 i].attr_weights_configs_mapping[KERNEL_ATTR])
        self.assertEqual(tanh_qco, sevenbit_qco)
        self.assertEqual(avg_pool2d_k2_qco, sixbit_qco)
        self.assertEqual(avg_pool2d_qco, default_qco)

    def test_get_layers_by_op(self):
        op_obj = schema.OperatorsSet(name='opsetA')

        hm = schema.TargetPlatformCapabilities(
            default_qco=schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC])),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            operator_set=tuple([op_obj]),
            add_metadata=False)

        fw_tp = FrameworkQuantizationCapabilities(hm)
        with fw_tp:
            opset_layers = [torch.nn.Conv2d, LayerFilterParams(torch.nn.Softmax, dim=1)]
            OperationsSetToLayers('opsetA', opset_layers)
        self.assertEqual(fw_tp.get_layers_by_opset_name('opsetA'), opset_layers)
        self.assertEqual(fw_tp.get_layers_by_opset(op_obj), opset_layers)

    def test_get_layers_by_opconcat(self):
        op_obj_a = schema.OperatorsSet(name='opsetA')
        op_obj_b = schema.OperatorsSet(name='opsetB')
        op_concat = schema.OperatorSetGroup(operators_set=[op_obj_a, op_obj_b])

        hm = schema.TargetPlatformCapabilities(
            default_qco=schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC])),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            operator_set=tuple([op_obj_a, op_obj_b]),
            add_metadata=False)

        fw_tp = FrameworkQuantizationCapabilities(hm)
        with fw_tp:
            opset_layers_a = [torch.nn.Conv2d]
            opset_layers_b = [LayerFilterParams(torch.nn.Softmax, dim=1)]
            OperationsSetToLayers('opsetA', opset_layers_a)
            OperationsSetToLayers('opsetB', opset_layers_b)

        self.assertEqual(fw_tp.get_layers_by_opset(op_concat), opset_layers_a + opset_layers_b)

    def test_layer_attached_to_multiple_opsets(self):
        hm = schema.TargetPlatformCapabilities(
            default_qco=schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC])),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            operator_set=tuple([
                schema.OperatorsSet(name='opsetA'),
                schema.OperatorsSet(name='opsetB')]),
            add_metadata=False)

        fw_tp = FrameworkQuantizationCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with fw_tp:
                OperationsSetToLayers('opsetA', [torch.nn.Conv2d])
                OperationsSetToLayers('opsetB', [torch.nn.Conv2d])
        self.assertEqual('Found layer Conv2d in more than one OperatorsSet', str(e.exception))

    def test_filter_layer_attached_to_multiple_opsets(self):
        hm = schema.TargetPlatformCapabilities(
            default_qco=schema.QuantizationConfigOptions(quantization_configurations=tuple([TEST_QC])),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            operator_set=tuple([schema.OperatorsSet(name='opsetA'),
                          schema.OperatorsSet(name='opsetB')]),
            add_metadata=False)

        fw_tp = FrameworkQuantizationCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with fw_tp:
                OperationsSetToLayers('opsetA', [LayerFilterParams(torch.nn.Softmax, dim=2)])
                OperationsSetToLayers('opsetB', [LayerFilterParams(torch.nn.Softmax, dim=2)])
        self.assertEqual('Found layer Softmax(dim=2) in more than one OperatorsSet', str(e.exception))

    def test_pytorch_fusing_patterns(self):
        default_qco = schema.QuantizationConfigOptions(quantization_configurations=tuple(
            [TEST_QC]))
        a = schema.OperatorsSet(name="opA")
        b = schema.OperatorsSet(name="opB")
        c = schema.OperatorsSet(name="opC")
        operator_set = [a, b, c]
        fusing_patterns = [schema.Fusing(operator_groups=(a, b, c)),
                           schema.Fusing(operator_groups=(a, c))]
        hm = schema.TargetPlatformCapabilities(default_qco=default_qco,
                                               tpc_minor_version=None,
                                               tpc_patch_version=None,
                                               tpc_platform_type=None,
                                               operator_set=tuple(operator_set),
                                               fusing_patterns=tuple(fusing_patterns),
                                               add_metadata=False)

        hm_torch = FrameworkQuantizationCapabilities(hm)
        with hm_torch:
            OperationsSetToLayers("opA", [torch.conv2d])
            OperationsSetToLayers("opB", [torch.tanh])
            OperationsSetToLayers("opC", [LayerFilterParams(torch.relu, Greater("max_value", 7), negative_slope=0)])

        fusings = hm_torch.get_fusing_patterns()
        self.assertEqual(len(fusings), 2)
        p0, p1 = fusings[0].get(FUSED_LAYER_PATTERN), fusings[1].get(FUSED_LAYER_PATTERN)

        self.assertEqual(len(p0), 3)
        self.assertEqual(p0[0], torch.conv2d)
        self.assertEqual(p0[1], torch.tanh)
        self.assertEqual(p0[2], LayerFilterParams(torch.relu, Greater("max_value", 7), negative_slope=0))

        self.assertEqual(len(p1), 2)
        self.assertEqual(p1[0], torch.conv2d)
        self.assertEqual(p1[1], LayerFilterParams(torch.relu, Greater("max_value", 7), negative_slope=0))


class TestGetPytorchTPC(unittest.TestCase):

    def test_get_pytorch_models(self):
        tpc = mct.get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)
        model = mobilenet_v2(pretrained=True)

        def rep_data():
            yield [np.random.randn(1, 3, 224, 224)]

        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(model,
                                                                        rep_data,
                                                                        target_platform_capabilities=tpc)

        mp_qc = MixedPrecisionQuantizationConfig()
        mp_qc.num_of_images = 1
        core_config = mct.core.CoreConfig(quantization_config=mct.core.QuantizationConfig(),
                                          mixed_precision_config=mp_qc)
        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(model,
                                                                        rep_data,
                                                                        target_resource_utilization=mct.core.ResourceUtilization(
                                                                            np.inf),
                                                                        target_platform_capabilities=tpc,
                                                                        core_config=core_config)

    def test_get_pytorch_supported_version(self):
        tpc = mct.get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)  # Latest
        self.assertTrue(tpc.tpc_minor_version == 1)

        tpc = mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, "v1")
        self.assertTrue(tpc.tpc_minor_version == 1)

        tpc = mct.get_target_platform_capabilities(PYTORCH, TFLITE_TP_MODEL, "v1")
        self.assertTrue(tpc.tpc_minor_version == 1)

        tpc = mct.get_target_platform_capabilities(PYTORCH, QNNPACK_TP_MODEL, "v1")
        self.assertTrue(tpc.tpc_minor_version == 1)

    def test_get_pytorch_not_supported_platform(self):
        with self.assertRaises(Exception) as e:
            mct.get_target_platform_capabilities(PYTORCH, "platform1")
        self.assertTrue(e.exception)

    def test_get_pytorch_not_supported_fw(self):
        with self.assertRaises(Exception) as e:
            mct.get_target_platform_capabilities("ONNX", DEFAULT_TP_MODEL)
        self.assertTrue(e.exception)

    def test_get_pytorch_not_supported_version(self):
        with self.assertRaises(Exception) as e:
            mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, "v0")
        self.assertTrue(e.exception)


def get_node(layer) -> BaseNode:
    model = LayerTestModel(layer)

    def rep_data():
        yield [np.random.randn(1, 3, 16, 16)]

    graph = PytorchImplementation().model_reader(model, rep_data)
    return graph.get_topo_sorted_nodes()[1]
