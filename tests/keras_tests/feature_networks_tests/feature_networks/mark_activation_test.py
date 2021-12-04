# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


from tests.keras_tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import tensorflow as tf

keras = tf.keras
layers = keras.layers


class MarkActivationTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test, kernel_op_layer, activation_function):
        assert kernel_op_layer in [layers.Conv2D,
                                   layers.DepthwiseConv2D,
                                   layers.Dense,
                                   layers.Conv2DTranspose], f'layer {kernel_op_layer} not in substitution'
        self.activation_function = activation_function
        self.kernel_op_layer = kernel_op_layer
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        if self.kernel_op_layer is layers.Dense:
            outputs = self.activation_function(self.kernel_op_layer(3)(inputs))
        else:
            outputs = self.activation_function(self.kernel_op_layer(3, 4)(inputs))

        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        kernel_op_layer_index = 2
        self.unit_test.assertTrue(isinstance(quantized_model.layers[kernel_op_layer_index], self.kernel_op_layer))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[kernel_op_layer_index + 1], (layers.ReLU,
                                                                                                 layers.Activation,
                                                                                                 layers.PReLU)))
