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


class CustomIdentity(keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs



class TestUnsupportedCustomLayer(unittest.TestCase):

    def test_raised_error_with_custom_layer(self):
        inputs = layers.Input(shape=(3, 3, 3))
        x = CustomIdentity()(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        expected_error = f'MCT does not support optimizing Keras custom layers, but found layer of type <class ' \
                         f'\'test_unsupported_custom_layer.CustomIdentity\'>. Please file a feature request or an issue if ' \
                         f'you believe this is an issue.'

        with self.assertRaises(Exception) as e:
            mct.ptq.keras_post_training_quantization_experimental(model,
                                                                  lambda _: [np.random.randn(1, 3, 3, 3)])
        # Remove class object path to compare with expected error message
        err_msg = str(e.exception)
        err_msg = err_msg[:err_msg.find('<class ')+8] + err_msg[err_msg.find('test_unsupported_custom_layer'):]
        self.assertEqual(expected_error, err_msg)
