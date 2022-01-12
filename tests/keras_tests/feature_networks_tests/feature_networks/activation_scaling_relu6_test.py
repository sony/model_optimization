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
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest


import model_compression_toolkit as mct
import tensorflow as tf
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers


class ActivationScalingReLU6Test(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        qc = super(ActivationScalingReLU6Test, self).get_quantization_config()
        qc.relu_unbound_correction = True
        return qc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Dense(20)(inputs)
        x = layers.ReLU(max_value=6)(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        alpha = (quantized_model.layers[2].weights[0] / float_model.layers[1].weights[0]).numpy().mean()
        beta = (quantized_model.layers[6].weights[0] / float_model.layers[3].weights[0]).numpy().mean()

        self.unit_test.assertTrue(np.allclose(alpha, 8 / 6, atol=1e-1))
        self.unit_test.assertTrue(np.allclose(beta, 6 / 8, atol=1e-1))
        self.unit_test.assertTrue(quantized_model.layers[4].max_value is None)
