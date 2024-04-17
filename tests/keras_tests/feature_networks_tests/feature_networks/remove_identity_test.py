# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import keras
import tensorflow as tf

class RemoveIdentityTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = keras.layers.Input(shape=self.get_input_shapes()[0][1:])
        x = keras.layers.Conv2D(3, 3)(inputs)
        x = keras.layers.Identity()(x)
        x = tf.identity(x)
        outputs = keras.layers.BatchNormalization()(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # Make sure identity and bn layers are not in the final model.
        # there should be 4 layers: input, input_quantizer, conv, conv_quantizer
        self.unit_test.assertTrue(len(quantized_model.layers)==4)

