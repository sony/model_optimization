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

import numpy as np
import torch
from torch.nn import Hardtanh
from torch.nn.functional import hardtanh
from torchvision.models import mobilenet_v2

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.constants import PYTORCH
from model_compression_toolkit.core.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.core.common.target_platform.targetplatform2framework import LayerFilterParams
from model_compression_toolkit.core.common.target_platform.targetplatform2framework.attribute_filter import Greater, \
    Smaller, Eq
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.core.pytorch.constants import DEFAULT_TP_MODEL, TFLITE_TP_MODEL, QNNPACK_TP_MODEL
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from tests.common_tests.test_tp_model import TEST_QC, TEST_QCO
from tests.pytorch_tests.layer_tests.base_pytorch_layer_test import LayerTestModel

tp = mct.target_platform


class TestPytorchTPModel(unittest.TestCase):

    def test_pytorch_layers_with_params(self):
        hardtanh_with_params = LayerFilterParams(Hardtanh, Greater("max_val", 2))
        self.assertTrue(hardtanh_with_params.match(get_node(Hardtanh(max_val=3))))
        self.assertFalse(hardtanh_with_params.match(get_node(Hardtanh(max_val=2))))
        self.assertFalse(hardtanh_with_params.match(get_node(Hardtanh(max_val=1))))

        hardtanh_with_params = LayerFilterParams(hardtanh, Greater("max_val", 2))
        self.assertTrue(hardtanh_with_params.match(get_node(partial(hardtanh, max_val=3))))
        self.assertFalse(hardtanh_with_params.match(get_node(partial(hardtanh, max_val=2))))
        self.assertFalse(hardtanh_with_params.match(get_node(partial(hardtanh, max_val=1))))

        hardtanh_with_params = LayerFilterParams(Hardtanh, Greater("max_val", 2) & Smaller("min_val", 1))
        self.assertTrue(hardtanh_with_params.match(get_node(Hardtanh(max_val=3, min_val=0))))
        self.assertFalse(hardtanh_with_params.match(get_node(Hardtanh(max_val=3, min_val=1))))
        self.assertFalse(hardtanh_with_params.match(get_node(Hardtanh(max_val=2, min_val=0.5))))
        self.assertFalse(hardtanh_with_params.match(get_node(Hardtanh(max_val=2))))
        self.assertFalse(hardtanh_with_params.match(get_node(Hardtanh(max_val=1, min_val=0.5))))

        hardtanh_with_params = LayerFilterParams(hardtanh, Greater("max_val", 2) & Smaller("min_val", 1))
        self.assertTrue(hardtanh_with_params.match(get_node(partial(hardtanh, max_val=3, min_val=0))))
        self.assertFalse(hardtanh_with_params.match(get_node(partial(hardtanh, max_val=3, min_val=1))))
        self.assertFalse(hardtanh_with_params.match(get_node(partial(hardtanh, max_val=2, min_val=0.5))))
        self.assertFalse(hardtanh_with_params.match(get_node(partial(hardtanh, max_val=2))))
        self.assertFalse(hardtanh_with_params.match(get_node(partial(hardtanh, max_val=1, min_val=0.5))))

        l2norm_tflite_opset = LayerFilterParams(torch.nn.functional.normalize, Eq('p',2) | Eq('p',None))
        self.assertTrue(l2norm_tflite_opset.match(get_node(partial(torch.nn.functional.normalize, p=2))))
        self.assertTrue(l2norm_tflite_opset.match(get_node(partial(torch.nn.functional.normalize, p=2.0))))
        self.assertTrue(l2norm_tflite_opset.match(get_node(torch.nn.functional.normalize)))
        self.assertFalse(l2norm_tflite_opset.match(get_node(partial(torch.nn.functional.normalize, p=3.0))))



    def test_qco_by_pytorch_layer(self):
        default_qco = tp.QuantizationConfigOptions([TEST_QC])
        hm = tp.TargetPlatformModel(default_qco, name='test')
        with hm:
            mixed_precision_configuration_options = tp.QuantizationConfigOptions([TEST_QC,
                                                                                   TEST_QC.clone_and_edit(
                                                                                       weights_n_bits=4),
                                                                                   TEST_QC.clone_and_edit(
                                                                                       weights_n_bits=2)],
                                                                                  base_config=TEST_QC)

            tp.OperatorsSet("conv", mixed_precision_configuration_options)

            sevenbit_qco = TEST_QCO.clone_and_edit(activation_n_bits=7)
            tp.OperatorsSet("tanh", sevenbit_qco)

            sixbit_qco = TEST_QCO.clone_and_edit(activation_n_bits=6)
            tp.OperatorsSet("avg_pool2d_kernel_2", sixbit_qco)

            tp.OperatorsSet("avg_pool2d")

        hm_pytorch = tp.TargetPlatformCapabilities(hm, name='fw_test')
        with hm_pytorch:
            tp.OperationsSetToLayers("conv", [torch.nn.Conv2d])
            tp.OperationsSetToLayers("tanh", [torch.tanh])
            tp.OperationsSetToLayers("avg_pool2d_kernel_2",
                                      [LayerFilterParams(torch.nn.functional.avg_pool2d, kernel_size=2)])
            tp.OperationsSetToLayers("avg_pool2d",
                                      [torch.nn.functional.avg_pool2d])

        conv_node = get_node(torch.nn.Conv2d(3, 3, (1, 1)))
        tanh_node = get_node(torch.tanh)
        avg_pool2d_k2 = get_node(partial(torch.nn.functional.avg_pool2d, kernel_size=2))
        avg_pool2d = get_node(partial(torch.nn.functional.avg_pool2d, kernel_size=1))

        conv_qco = hm_pytorch.get_qco_by_node(conv_node)
        tanh_qco = hm_pytorch.get_qco_by_node(tanh_node)
        avg_pool2d_k2_qco = hm_pytorch.get_qco_by_node(avg_pool2d_k2)
        avg_pool2d_qco = hm_pytorch.get_qco_by_node(avg_pool2d)

        self.assertEqual(conv_qco, mixed_precision_configuration_options)
        self.assertEqual(tanh_qco, sevenbit_qco)
        self.assertEqual(avg_pool2d_k2_qco, sixbit_qco)
        self.assertEqual(avg_pool2d_qco, default_qco)

    def test_get_layers_by_op(self):
        hm = tp.TargetPlatformModel(tp.QuantizationConfigOptions([TEST_QC]))
        with hm:
            op_obj = tp.OperatorsSet('opsetA')
        fw_tp = TargetPlatformCapabilities(hm)
        with fw_tp:
            opset_layers = [torch.nn.Conv2d, LayerFilterParams(torch.nn.Softmax, dim=1)]
            tp.OperationsSetToLayers('opsetA', opset_layers)
        self.assertEqual(fw_tp.get_layers_by_opset_name('opsetA'), opset_layers)
        self.assertEqual(fw_tp.get_layers_by_opset(op_obj), opset_layers)

    def test_get_layers_by_opconcat(self):
        hm = tp.TargetPlatformModel(tp.QuantizationConfigOptions([TEST_QC]))
        with hm:
            op_obj_a = tp.OperatorsSet('opsetA')
            op_obj_b = tp.OperatorsSet('opsetB')
            op_concat = tp.OperatorSetConcat(op_obj_a, op_obj_b)

        fw_tp = TargetPlatformCapabilities(hm)
        with fw_tp:
            opset_layers_a = [torch.nn.Conv2d]
            opset_layers_b = [LayerFilterParams(torch.nn.Softmax, dim=1)]
            tp.OperationsSetToLayers('opsetA', opset_layers_a)
            tp.OperationsSetToLayers('opsetB', opset_layers_b)

        self.assertEqual(fw_tp.get_layers_by_opset_name('opsetA_opsetB'), opset_layers_a + opset_layers_b)
        self.assertEqual(fw_tp.get_layers_by_opset(op_concat), opset_layers_a + opset_layers_b)

    def test_layer_attached_to_multiple_opsets(self):
        hm = tp.TargetPlatformModel(tp.QuantizationConfigOptions([TEST_QC]))
        with hm:
            tp.OperatorsSet('opsetA')
            tp.OperatorsSet('opsetB')

        fw_tp = TargetPlatformCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with fw_tp:
                tp.OperationsSetToLayers('opsetA', [torch.nn.Conv2d])
                tp.OperationsSetToLayers('opsetB', [torch.nn.Conv2d])
        self.assertEqual('Found layer Conv2d in more than one OperatorsSet', str(e.exception))

    def test_filter_layer_attached_to_multiple_opsets(self):
        hm = tp.TargetPlatformModel(tp.QuantizationConfigOptions([TEST_QC]))
        with hm:
            tp.OperatorsSet('opsetA')
            tp.OperatorsSet('opsetB')

        fw_tp = TargetPlatformCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with fw_tp:
                tp.OperationsSetToLayers('opsetA', [LayerFilterParams(torch.nn.Softmax, dim=2)])
                tp.OperationsSetToLayers('opsetB', [LayerFilterParams(torch.nn.Softmax, dim=2)])
        self.assertEqual('Found layer Softmax(dim=2) in more than one OperatorsSet', str(e.exception))

    def test_opset_not_in_tp(self):
        default_qco = tp.QuantizationConfigOptions([TEST_QC])
        hm = tp.TargetPlatformModel(default_qco)
        hm_pytorch = tp.TargetPlatformCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with hm_pytorch:
                tp.OperationsSetToLayers("conv", [torch.nn.Conv2d])
        self.assertEqual(
            'conv is not defined in the target platform model that is associated with the target platform capabilities.',
            str(e.exception))

    def test_pytorch_fusing_patterns(self):
        default_qco = tp.QuantizationConfigOptions([TEST_QC])
        hm = tp.TargetPlatformModel(default_qco)
        with hm:
            a = tp.OperatorsSet("opA")
            b = tp.OperatorsSet("opB")
            c = tp.OperatorsSet("opC")
            tp.Fusing([a, b, c])
            tp.Fusing([a, c])

        hm_keras = tp.TargetPlatformCapabilities(hm)
        with hm_keras:
            tp.OperationsSetToLayers("opA", [torch.conv2d])
            tp.OperationsSetToLayers("opB", [torch.tanh])
            tp.OperationsSetToLayers("opC", [LayerFilterParams(torch.relu, Greater("max_value", 7), negative_slope=0)])

        fusings = hm_keras.get_fusing_patterns()
        self.assertEqual(len(fusings), 2)
        p0, p1 = fusings[0], fusings[1]

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
            return [np.random.randn(1, 3, 224, 224)]

        quantized_model, _ = mct.pytorch_post_training_quantization(model,
                                                                    rep_data,
                                                                    n_iter=1,
                                                                    target_platform_capabilities=tpc)

        mp_qc = copy.deepcopy(DEFAULT_MIXEDPRECISION_CONFIG)
        mp_qc.num_of_images = 1
        quantized_model, _ = mct.pytorch_post_training_quantization_mixed_precision(model,
                                                                                    rep_data,
                                                                                    target_kpi=mct.KPI(np.inf),
                                                                                    n_iter=1,
                                                                                    target_platform_capabilities=tpc,
                                                                                    quant_config=mp_qc)

    def test_get_pytorch_supported_version(self):
        mct.get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)
        mct.get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL, "v1")
        mct.get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL, "v2")
        mct.get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL, "v3")
        mct.get_target_platform_capabilities(PYTORCH, TFLITE_TP_MODEL, "v1")
        mct.get_target_platform_capabilities(PYTORCH, QNNPACK_TP_MODEL, "v1")

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
            mct.get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL, "v0")
        self.assertTrue(e.exception)



def get_node(layer):
    model = LayerTestModel(layer)
    graph = PytorchImplementation().model_reader(model, lambda: [np.random.randn(1, 3, 16, 16)])
    return graph.get_topo_sorted_nodes()[1]
