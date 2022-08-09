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

keras = tf.keras
layers = keras.layers
tp = cmo.target_platform


class UniformRangeSelectionActivationTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, activation_threshold_method):
        super().__init__(unit_test)
        self.activation_threshold_method = activation_threshold_method

    def generate_inputs(self):
        return [np.random.uniform(low=-7, high=7, size=in_shape) for in_shape in self.get_input_shapes()]

    def get_quantization_config(self):
        return cmo.QuantizationConfig(activation_error_method=self.activation_threshold_method)

    def get_tpc(self):
        tp_model = generate_test_tp_model({
            'activation_quantization_method': tp.QuantizationMethod.UNIFORM,
            'activation_n_bits': 8})
        return generate_keras_tpc(name="uniform_range_test", tp_model=tp_model)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.ReLU()(inputs)
        outputs = tf.add(x, -1)  # to get negative values in activation to test signed symmetric quantization
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify quantization range contains zero
        fake_layer_input_args = quantized_model.layers[1].inbound_nodes[0].call_kwargs
        fake_layer_add_args = quantized_model.layers[5].inbound_nodes[0].call_kwargs

        input_layer_min, input_layer_max = fake_layer_input_args['min'], fake_layer_input_args['max']
        add_layer_min, add_layer_max = fake_layer_add_args['min'], fake_layer_add_args['max']

        self.unit_test.assertTrue(input_layer_min <= 0.0 <= input_layer_max,
                                  msg=f"0.0 is not within the quantization range ({input_layer_min}, {input_layer_max}) "
                                      f"for Input layer.")
        self.unit_test.assertTrue(add_layer_min <= 0.0 <= add_layer_max,
                                  msg=f"0.0 is not within the quantization range ({add_layer_min}, {add_layer_max}) "
                                      f"for Relu layer.")
