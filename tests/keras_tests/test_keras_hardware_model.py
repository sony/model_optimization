# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, Activation, Input
else:
    from keras.layers import Conv2D, Conv2DTranspose, ReLU, Activation
    from keras import Input

import model_compression_toolkit as mct
from model_compression_toolkit.common.constants import TENSORFLOW
from model_compression_toolkit.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.common.target_platform.targetplatform2framework import LayerFilterParams
from model_compression_toolkit.common.target_platform.targetplatform2framework.attribute_filter import Greater, \
    Smaller, GreaterEq, Eq, SmallerEq
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.keras.constants import DEFAULT_TP_MODEL, QNNPACK_TP_MODEL, TFLITE_TP_MODEL
from model_compression_toolkit.keras.keras_implementation import KerasImplementation
from tests.common_tests.test_hardware_model import TEST_QCO, TEST_QC

hwm = mct.target_platform

def get_node(layer):
    i = Input(shape=(3, 16, 16))
    x = layer(i)
    model = tf.keras.Model(i, x)
    graph = KerasImplementation().model_reader(model, None)
    return graph.get_topo_sorted_nodes()[1]


class TestKerasHWModel(unittest.TestCase):

    def test_keras_layers_with_params(self):
        conv_with_params = LayerFilterParams(Conv2D,
                                             Greater("filters", 2),
                                             Smaller("filters", 4),
                                             activation='softmax',
                                             kernel_size=(3, 4),
                                             filters=3)

        conv = Conv2D(filters=3, kernel_size=(3, 4), activation='softmax')
        self.assertTrue(conv_with_params.match(get_node(conv)))
        conv = Conv2D(filters=2, kernel_size=(3, 4), activation='softmax')
        self.assertFalse(conv_with_params.match(get_node(conv)))
        conv = Conv2DTranspose(filters=3, kernel_size=(3, 4), activation='softmax')
        self.assertFalse(conv_with_params.match(get_node(conv)))

        relu_with_params = LayerFilterParams(ReLU, GreaterEq("max_value", 0.5) | Smaller("max_value", 0.2))
        self.assertTrue(relu_with_params.match(get_node(ReLU(max_value=0.1))))
        self.assertTrue(relu_with_params.match(get_node(ReLU(max_value=0.5))))
        self.assertFalse(relu_with_params.match(get_node(ReLU(max_value=0.3))))

        relu_with_params = LayerFilterParams(ReLU, Eq("max_value", None) | Eq("max_value", 6))
        self.assertTrue(relu_with_params.match(get_node(ReLU())))
        self.assertTrue(relu_with_params.match(get_node(ReLU(max_value=6))))
        self.assertFalse(relu_with_params.match(get_node(ReLU(max_value=8))))

        lrelu_with_params = LayerFilterParams(tf.nn.leaky_relu, SmallerEq("alpha", 2))
        self.assertTrue(lrelu_with_params.match(get_node(partial(tf.nn.leaky_relu, alpha=0.4))))
        self.assertTrue(lrelu_with_params.match(get_node(partial(tf.nn.leaky_relu, alpha=2))))
        self.assertFalse(lrelu_with_params.match(get_node(partial(tf.nn.leaky_relu, alpha=2.1))))

        lrelu_with_params = LayerFilterParams(tf.nn.leaky_relu)
        self.assertTrue(lrelu_with_params.match(get_node(partial(tf.nn.leaky_relu, alpha=0.4))))

    def test_get_layers_by_op(self):
        hm = hwm.TargetPlatformModel(hwm.QuantizationConfigOptions([TEST_QC]))
        with hm:
            op_obj = hwm.OperatorsSet('opsetA')
        fw_hwm = TargetPlatformCapabilities(hm)
        with fw_hwm:
            opset_layers = [Conv2D, LayerFilterParams(ReLU, max_value=2)]
            hwm.OperationsSetToLayers('opsetA', opset_layers)
        self.assertEqual(fw_hwm.get_layers_by_opset_name('opsetA'), opset_layers)
        self.assertEqual(fw_hwm.get_layers_by_opset(op_obj), opset_layers)

    def test_get_layers_by_opconcat(self):
        hm = hwm.TargetPlatformModel(hwm.QuantizationConfigOptions([TEST_QC]))
        with hm:
            op_obj_a = hwm.OperatorsSet('opsetA')
            op_obj_b = hwm.OperatorsSet('opsetB')
            op_concat = hwm.OperatorSetConcat(op_obj_a, op_obj_b)

        fw_hwm = TargetPlatformCapabilities(hm)
        with fw_hwm:
            opset_layers_a = [Conv2D]
            opset_layers_b = [LayerFilterParams(ReLU, max_value=2)]
            hwm.OperationsSetToLayers('opsetA', opset_layers_a)
            hwm.OperationsSetToLayers('opsetB', opset_layers_b)

        self.assertEqual(fw_hwm.get_layers_by_opset_name('opsetA_opsetB'), opset_layers_a + opset_layers_b)
        self.assertEqual(fw_hwm.get_layers_by_opset(op_concat), opset_layers_a + opset_layers_b)

    def test_layer_attached_to_multiple_opsets(self):
        hm = hwm.TargetPlatformModel(hwm.QuantizationConfigOptions([TEST_QC]))
        with hm:
            hwm.OperatorsSet('opsetA')
            hwm.OperatorsSet('opsetB')

        fw_hwm = TargetPlatformCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with fw_hwm:
                hwm.OperationsSetToLayers('opsetA', [Conv2D])
                hwm.OperationsSetToLayers('opsetB', [Conv2D])
        self.assertEqual('Found layer Conv2D in more than one OperatorsSet', str(e.exception))

    def test_filter_layer_attached_to_multiple_opsets(self):
        hm = hwm.TargetPlatformModel(hwm.QuantizationConfigOptions([TEST_QC]))
        with hm:
            hwm.OperatorsSet('opsetA')
            hwm.OperatorsSet('opsetB')

        fw_hwm = TargetPlatformCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with fw_hwm:
                hwm.OperationsSetToLayers('opsetA', [LayerFilterParams(Activation, activation="relu")])
                hwm.OperationsSetToLayers('opsetB', [LayerFilterParams(Activation, activation="relu")])
        self.assertEqual('Found layer Activation(activation=relu) in more than one OperatorsSet', str(e.exception))

    def test_qco_by_keras_layer(self):
        default_qco = hwm.QuantizationConfigOptions([TEST_QC])
        hm = hwm.TargetPlatformModel(default_qco, name='test')
        with hm:
            mixed_precision_configuration_options = hwm.QuantizationConfigOptions([TEST_QC,
                                                                                   TEST_QC.clone_and_edit(
                                                                                       weights_n_bits=4),
                                                                                   TEST_QC.clone_and_edit(
                                                                                       weights_n_bits=2)],
                                                                                  base_config=TEST_QC)

            hwm.OperatorsSet("conv", mixed_precision_configuration_options)
            sevenbit_qco = TEST_QCO.clone_and_edit(activation_n_bits=7)
            hwm.OperatorsSet("tanh", sevenbit_qco)
            hwm.OperatorsSet("relu")

        hm_keras = hwm.TargetPlatformCapabilities(hm, name='fw_test')
        with hm_keras:
            hwm.OperationsSetToLayers("conv", [Conv2D])
            hwm.OperationsSetToLayers("tanh", [tf.nn.tanh])
            hwm.OperationsSetToLayers("relu", [LayerFilterParams(Activation, activation="relu")])

        conv_node = get_node(Conv2D(1, 1))
        tanh_node = get_node(tf.nn.tanh)
        relu_node = get_node(Activation('relu'))

        conv_qco = hm_keras.get_qco_by_node(conv_node)
        tanh_qco = hm_keras.get_qco_by_node(tanh_node)
        relu_qco = hm_keras.get_qco_by_node(relu_node)

        self.assertEqual(conv_qco, mixed_precision_configuration_options)
        self.assertEqual(tanh_qco, sevenbit_qco)
        self.assertEqual(relu_qco, default_qco)

    def test_opset_not_in_hwm(self):
        default_qco = hwm.QuantizationConfigOptions([TEST_QC])
        hm = hwm.TargetPlatformModel(default_qco)
        hm_keras = hwm.TargetPlatformCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with hm_keras:
                hwm.OperationsSetToLayers("conv", [Conv2D])
        self.assertEqual(
            'conv is not defined in the hardware model that is associated with the framework hardware model.',
            str(e.exception))

    def test_keras_fusing_patterns(self):
        default_qco = hwm.QuantizationConfigOptions([TEST_QC])
        hm = hwm.TargetPlatformModel(default_qco)
        with hm:
            a = hwm.OperatorsSet("opA")
            b = hwm.OperatorsSet("opB")
            c = hwm.OperatorsSet("opC")
            hwm.Fusing([a, b, c])
            hwm.Fusing([a, c])

        hm_keras = hwm.TargetPlatformCapabilities(hm)
        with hm_keras:
            hwm.OperationsSetToLayers("opA", [Conv2D])
            hwm.OperationsSetToLayers("opB", [tf.nn.tanh])
            hwm.OperationsSetToLayers("opC", [LayerFilterParams(ReLU, Greater("max_value", 7), negative_slope=0)])

        fusings = hm_keras.get_fusing_patterns()
        self.assertEqual(len(fusings), 2)
        p0, p1 = fusings[0], fusings[1]

        self.assertEqual(len(p0), 3)
        self.assertEqual(p0[0], Conv2D)
        self.assertEqual(p0[1], tf.nn.tanh)
        self.assertEqual(p0[2], LayerFilterParams(ReLU, Greater("max_value", 7), negative_slope=0))

        self.assertEqual(len(p1), 2)
        self.assertEqual(p1[0], Conv2D)
        self.assertEqual(p1[1], LayerFilterParams(ReLU, Greater("max_value", 7), negative_slope=0))


class TestGetKerasHardwareModelAPI(unittest.TestCase):
    def test_get_keras_hw_models(self):
        fw_hw_model = mct.get_model(TENSORFLOW, DEFAULT_TP_MODEL)
        model = MobileNetV2()

        def rep_data():
            return [np.random.randn(1, 224, 224, 3)]

        quantized_model, _ = mct.keras_post_training_quantization(model,
                                                                  rep_data,
                                                                  n_iter=1,
                                                                  fw_hw_model=fw_hw_model)

        mp_qc = copy.deepcopy(DEFAULT_MIXEDPRECISION_CONFIG)
        mp_qc.num_of_images = 1
        quantized_model, _ = mct.keras_post_training_quantization_mixed_precision(model,
                                                                                  rep_data,
                                                                                  target_kpi=mct.KPI(np.inf),
                                                                                  n_iter=1,
                                                                                  quant_config=mp_qc,
                                                                                  fw_hw_model=fw_hw_model)

    def test_get_keras_not_supported_model(self):
        with self.assertRaises(Exception) as e:
            mct.get_model(TENSORFLOW, 'should_not_support')
        self.assertEqual('Hardware model named should_not_support is not supported for framework tensorflow', str(e.exception))
