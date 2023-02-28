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

import numpy as np
import tensorflow as tf

from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers
keras_softmax = tf.keras.activations.softmax


class SoftmaxShiftTest(BaseKerasFeatureNetworkTest):
    """
    This test checks the Softmax shift feature.
    """

    def __init__(self, unit_test, kernel_op_layer, activation_function):
        self.activation_function = activation_function
        self.kernel_op_layer = kernel_op_layer
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        if self.kernel_op_layer.__getattribute__('activation') == keras_softmax:
            outputs = self.kernel_op_layer(inputs)
            return keras.Model(inputs=inputs, outputs=outputs)
        else:
            softmax_outputs = self.activation_function(self.kernel_op_layer(inputs))
            dense_outputs = self.kernel_op_layer(inputs)
            return keras.Model(inputs=inputs, outputs=[dense_outputs, softmax_outputs])

    def get_quantization_config(self):
        qc = super(SoftmaxShiftTest, self).get_quantization_config()
        qc.softmax_shift = True
        return qc

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        if self.kernel_op_layer.__getattribute__('activation') == keras_softmax:
            quant_bias = quantized_model.layers[-4].bias
            float_bias = float_model.layers[-1].bias
            diff_bias = float_bias - quant_bias
            mean_diff_bias = np.mean(diff_bias)
            self.unit_test.assertTrue(np.allclose(diff_bias, mean_diff_bias, atol=1e-1))
        else:
            quant_bias = quantized_model.layers[-5].bias
            float_bias = float_model.layers[1].bias
            diff_bias = float_bias - quant_bias
            mean_diff_bias = np.mean(diff_bias)
            self.unit_test.assertTrue(np.allclose(diff_bias, mean_diff_bias, atol=1e-1))
            before_softmax_quantized_output = quantized_model(input_x)[0]
            before_softmax_float_output = float_model(input_x)[0]
            diff_output = before_softmax_float_output - before_softmax_quantized_output
            mean_diff_before_softmax = np.mean(diff_output)
            self.unit_test.assertTrue(np.allclose(mean_diff_before_softmax, mean_diff_bias, atol=1e-1))
