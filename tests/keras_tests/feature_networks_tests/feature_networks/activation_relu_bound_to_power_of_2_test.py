# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit import QuantizationConfig, QuantizationErrorMethod
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest

import tensorflow as tf
import numpy as np
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers


class ReLUBoundToPOTNetTest(BaseKerasFeatureNetworkTest):
    """
    This test checks the ReLU Bound To POT feature.
    """
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        qc = QuantizationConfig(QuantizationErrorMethod.MSE,
                                QuantizationErrorMethod.MSE,
                                relu_bound_to_power_of_2=True)
        return qc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Dense(20)(inputs)
        x = layers.ReLU(max_value=6)(x)
        x = layers.Dense(20)(x)
        x = layers.Dense(20)(x)
        x = layers.ReLU(max_value=17)(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        alpha_1 = (quantized_model.layers[2].weights[0] / float_model.layers[1].weights[0]).numpy().mean()
        beta_1 = (quantized_model.layers[5].weights[0] / float_model.layers[3].weights[0]).numpy().mean()
        alpha_2 = (quantized_model.layers[7].weights[0] / float_model.layers[4].weights[0]).numpy().mean()
        beta_2 = (quantized_model.layers[10].weights[0] / float_model.layers[6].weights[0]).numpy().mean()

        self.unit_test.assertTrue(np.allclose(alpha_1 * beta_1, 1, atol=1e-1))
        self.unit_test.assertTrue(np.allclose(alpha_1 * 6 / 8, 1, atol=1e-1))
        self.unit_test.assertTrue(np.allclose(8 / 6 * beta_1, 1, atol=1e-1))
        self.unit_test.assertTrue(np.allclose(alpha_2 * beta_2, 1, atol=1e-1))
        self.unit_test.assertTrue(np.allclose(alpha_2 * 17 / 8, 1, atol=1e-1))
        self.unit_test.assertTrue(np.allclose(8 / 17 * beta_2, 1, atol=1e-1))
