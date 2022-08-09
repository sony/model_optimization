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


import tensorflow as tf

from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers


class SplitConcatenateTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Dense(30)(inputs)
        output_split = tf.split(x, num_or_size_splits=30, axis=-1)
        y = layers.Conv2D(3, 4)(output_split[1])
        z = layers.Conv2D(3, 4)(output_split[3])
        outputs = layers.Concatenate()([y, z])
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(quantized_model.layers[4].function == tf.split)
        self.unit_test.assertTrue(isinstance(quantized_model.layers[5], layers.Conv2D))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[6], layers.Conv2D))
        self.unit_test.assertTrue(quantized_model.layers[7].function == tf.quantization.fake_quant_with_min_max_vars)
        self.unit_test.assertTrue(quantized_model.layers[8].function == tf.quantization.fake_quant_with_min_max_vars)

        self.unit_test.assertTrue(quantized_model.layers[4].output[1].ref() == quantized_model.layers[5].input.ref())
        self.unit_test.assertTrue(quantized_model.layers[4].output[3].ref() == quantized_model.layers[6].input.ref())

        self.unit_test.assertTrue(isinstance(quantized_model.layers[9], layers.Concatenate))
        self.unit_test.assertTrue(len(quantized_model.layers[9].input) == 2)
        self.unit_test.assertTrue(quantized_model.layers[9].input[0].ref() == quantized_model.layers[7].output.ref())
        self.unit_test.assertTrue(quantized_model.layers[9].input[1].ref() == quantized_model.layers[8].output.ref())
