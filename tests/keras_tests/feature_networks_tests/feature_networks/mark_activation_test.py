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


import tensorflow as tf

from model_compression_toolkit.keras.back2framework.model_builder import is_layer_fake_quant
from model_compression_toolkit.keras.constants import ACTIVATION

if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers.core import TFOpLambda
else:
    from keras.layers.core import TFOpLambda
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers


class MarkActivationTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, kernel_op_layer, activation_function):
        assert kernel_op_layer in [layers.Conv2D,
                                   layers.DepthwiseConv2D], f'layer {kernel_op_layer} not in substitution'
        self.activation_function = activation_function
        self.kernel_op_layer = kernel_op_layer
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = self.activation_function(self.kernel_op_layer(3, 4)(inputs))
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        kernel_op_layer_index = 2
        self.unit_test.assertTrue(isinstance(quantized_model.layers[kernel_op_layer_index], self.kernel_op_layer))
        activation_layer = quantized_model.layers[kernel_op_layer_index + 1]
        if isinstance(activation_layer, TFOpLambda):
            self.unit_test.assertTrue(activation_layer.function is self.activation_function)
        else:
            self.unit_test.assertTrue(isinstance(activation_layer, (layers.ReLU,
                                                                    layers.Activation,
                                                                    layers.PReLU,
                                                                    layers.ELU)))


class AssertNoMarkActivationTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, kernel_op_layer, activation_function):
        self.activation_function = activation_function
        self.kernel_op_layer = kernel_op_layer
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        if self.kernel_op_layer is layers.Dense:
            outputs = self.activation_function(self.kernel_op_layer(3)(inputs))
        else:
            outputs = self.activation_function(self.kernel_op_layer(3, 4)(inputs))
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        kernel_op_layer_index = 2
        self.unit_test.assertTrue(isinstance(quantized_model.layers[kernel_op_layer_index], self.kernel_op_layer))
        fq_layer = quantized_model.layers[kernel_op_layer_index + 1]
        self.unit_test.assertTrue(is_layer_fake_quant(fq_layer))
        activation_layer = quantized_model.layers[kernel_op_layer_index + 2]
        if isinstance(activation_layer, TFOpLambda):
            self.unit_test.assertTrue(activation_layer.function is self.activation_function)
        else:
            self.unit_test.assertTrue(activation_layer.get_config().get(ACTIVATION)==self.activation_function.get_config().get(ACTIVATION))