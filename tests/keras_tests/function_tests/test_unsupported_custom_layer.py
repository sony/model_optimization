# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
import unittest

import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct

keras = tf.keras
layers = keras.layers


class CustomFC(keras.layers.Layer):

    def __init__(self, units=32, input_dim=3):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class TestUnsupportedCustomLayer(unittest.TestCase):

    def test_raised_error_with_custom_layer(self):
        inputs = layers.Input(shape=(3, 3, 3))
        x = CustomFC()(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        expected_error = f'MCT does not support optimizing Keras custom layers, but found layer of type <class ' \
                         f'\'test_unsupported_custom_layer.CustomFC\'>. Please file a feature request or an issue if ' \
                         f'you believe this is an issue.'

        with self.assertRaises(Exception) as e:
            mct.ptq.keras_post_training_quantization_experimental(model,
                                                                  lambda _: [np.random.randn(1, 3, 3, 3)])
        self.assertEqual(expected_error, str(e.exception))
