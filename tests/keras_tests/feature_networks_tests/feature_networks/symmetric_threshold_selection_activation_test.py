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

from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as cmo

tp = cmo.target_platform
keras = tf.keras
layers = keras.layers


class SymmetricThresholdSelectionActivationTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, activation_threshold_method):
        super().__init__(unit_test)
        self.activation_threshold_method = activation_threshold_method

    def generate_inputs(self):
        return [np.random.uniform(low=-7, high=7, size=in_shape) for in_shape in self.get_input_shapes()]

    def get_tpc(self):
        tp_model = generate_test_tp_model({
            'activation_quantization_method': tp.QuantizationMethod.SYMMETRIC,
            'activation_n_bits': 8})
        return generate_keras_tpc(name="symmetric_threshold_test", tp_model=tp_model)

    def get_quantization_config(self):
        return cmo.QuantizationConfig(activation_error_method=self.activation_threshold_method)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.ReLU()(inputs)
        outputs = tf.add(x, -1)  # to get negative values in activation to test signed symmetric quantization
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify threshold not power of 2 and unsigned symmetric range for first two layers' activations
        fake_layer_input_args = quantized_model.layers[1].inbound_nodes[0].call_kwargs
        fake_layer_relu_args = quantized_model.layers[3].inbound_nodes[0].call_kwargs

        threshold_input = fake_layer_input_args['max'] / (2 ** fake_layer_input_args['num_bits'] - 1) \
                          + fake_layer_input_args['max']
        threshold_relu = fake_layer_relu_args['max'] / (2 ** fake_layer_relu_args['num_bits'] - 1) \
                         + fake_layer_relu_args['max']

        self.unit_test.assertFalse(
            np.log2(threshold_input).is_integer(), msg=f"Input layer threshold {threshold_input} is a power of 2")
        self.unit_test.assertFalse(
            np.log2(threshold_relu).is_integer(), msg=f"ReLU layer threshold {threshold_relu} is a power of 2")

        self.unit_test.assertTrue(fake_layer_relu_args['min'] == 0,
                                  msg=f"Lower bound of ReLU layer unsigned symmetric range is "
                                      f"{fake_layer_relu_args['min']} (expected: 0)")

        # verify threshold not power of 2 and signed symmetric range for first Add activation layer
        fake_layer_add_args = quantized_model.layers[5].inbound_nodes[0].call_kwargs

        threshold_add = fake_layer_add_args['max'] / (2 ** fake_layer_add_args['num_bits'] - 1) \
                         + fake_layer_add_args['max']
        self.unit_test.assertFalse(
            np.log2(threshold_input).is_integer(), msg=f"Add layer threshold {threshold_add} is a power of 2")
        self.unit_test.assertTrue(fake_layer_add_args['min'] < 0,
                                  msg=f"Lower bound of Add layer signed symmetric range "
                                      f"{fake_layer_add_args['min']} is not negative")
