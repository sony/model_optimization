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


import tensorflow as tf
import numpy as np
from keras.engine.base_layer import Layer
from keras.layers import TFOpLambda

from mct_quantizers import KerasActivationQuantizationHolder
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as mct
from tests.keras_tests.utils import get_layers_from_model_by_type

tp = mct.target_platform
keras = tf.keras
layers = keras.layers


class SymmetricThresholdSelectionActivationTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, activation_threshold_method):
        super().__init__(unit_test, experimental_exporter=True)
        self.activation_threshold_method = activation_threshold_method

    def generate_inputs(self):
        return [np.random.uniform(low=-7, high=7, size=in_shape) for in_shape in self.get_input_shapes()]

    def get_tpc(self):
        tp_model = generate_test_tp_model({
            'activation_quantization_method': tp.QuantizationMethod.SYMMETRIC,
            'activation_n_bits': 8})
        return generate_keras_tpc(name="symmetric_threshold_test", tp_model=tp_model)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(activation_error_method=self.activation_threshold_method)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.ReLU()(inputs)
        outputs = tf.add(x, -1)  # to get negative values in activation to test signed symmetric quantization
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify threshold not power of 2 and unsigned symmetric range for first two layers' activations
        input_holder_layer = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[0]
        fake_layer_input_args = input_holder_layer.activation_holder_quantizer.get_config()
        relu_holder_layer = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1]
        fake_layer_relu_args = relu_holder_layer.activation_holder_quantizer.get_config()

        threshold_input = fake_layer_input_args['threshold'][0]
        threshold_relu = fake_layer_relu_args['threshold'][0]

        self.unit_test.assertFalse(
            np.log2(threshold_input).is_integer(), msg=f"Input layer threshold {threshold_input} is a power of 2")
        self.unit_test.assertFalse(
            np.log2(threshold_relu).is_integer(), msg=f"ReLU layer threshold {threshold_relu} is a power of 2")

        self.unit_test.assertTrue(fake_layer_relu_args['signed'] == False,
                                  msg=f"ReLU expected to have unsigned symmetric quantization param but is signed")

        # verify threshold not power of 2 and signed symmetric range for first Add activation layer
        add_holder_layer = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[2]
        fake_layer_add_args = add_holder_layer.activation_holder_quantizer.get_config()

        threshold_add = fake_layer_add_args['threshold'][0]

        self.unit_test.assertFalse(
            np.log2(threshold_input).is_integer(), msg=f"Add layer threshold {threshold_add} is a power of 2")
        self.unit_test.assertTrue(fake_layer_add_args['signed'] == True,
                                  msg=f"Add expected to have signed quantization range but is unsigned")


class SymmetricThresholdSelectionBoundedActivationTest(SymmetricThresholdSelectionActivationTest):
    def __init__(self, unit_test, activation_threshold_method):
        super().__init__(unit_test, activation_threshold_method)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Softmax()(inputs)
        outputs = tf.add(x, 1)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        fake_layer_input_args = holder_layers[0].activation_holder_quantizer.get_config()
        fake_layer_softmax_args = holder_layers[1].activation_holder_quantizer.get_config()

        threshold_input = fake_layer_input_args['threshold'][0]
        threshold_softmax = fake_layer_softmax_args['threshold'][0]

        # Verify threshold not power of 2
        self.unit_test.assertFalse(
            np.log2(threshold_input).is_integer(), msg=f"Input layer threshold {threshold_input} is a power of 2")

        # Verify threshold is 1
        self.unit_test.assertTrue(threshold_softmax == 1.0,
                                  msg=f"Threshold of Softmax layer is {threshold_softmax} (expected: 1)")

        # Verify min/max is bounded by 0 and 1
        self.unit_test.assertTrue(fake_layer_softmax_args['signed'] == False,
                                  msg=f"Softmax layer symmetric range is signed. Expected to be unsigned")
        self.unit_test.assertTrue(fake_layer_softmax_args['threshold'] == 1.0,
                                  msg=f"Softmax layer threshold is {fake_layer_softmax_args['threshold']}. Expected to be 1")
