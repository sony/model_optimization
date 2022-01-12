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
import numpy as np

from model_compression_toolkit.keras.constants import ACTIVATION, LINEAR
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as mct

keras = tf.keras
layers = keras.layers


class ActivationDecompositionTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, activation_function: str):
        self.activation_function = activation_function
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.QuantizationConfig(enable_activation_quantization=False, enable_weights_quantization=False)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = layers.Conv2D(3, 4, activation=self.activation_function)(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1], layers.Conv2D))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2], layers.Activation))
        self.unit_test.assertTrue(
            quantized_model.layers[1].get_config().get(ACTIVATION) == LINEAR)
        self.unit_test.assertTrue(
            quantized_model.layers[2].get_config().get(ACTIVATION) == self.activation_function)

