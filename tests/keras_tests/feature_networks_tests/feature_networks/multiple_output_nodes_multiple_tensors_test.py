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

if tf.__version__ < "2.6":
    from tensorflow.python.keras.engine.functional import Functional
    from tensorflow.python.keras.engine.sequential import Sequential
else:
    from keras.models import Functional, Sequential

import model_compression_toolkit as mct
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class MultipleOutputNodesMultipleTensors(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test,
                         input_shape=(20,20,3)) # Increase shape as the test has many convolutions

    def inner_functional_model(self, input_shape):
        inputs = layers.Input(shape=input_shape[1:])
        x = layers.Conv2D(3, 4)(inputs)
        y = layers.Conv2D(4, 5)(inputs)
        z = layers.Conv2D(5, 6)(inputs)
        w = layers.Conv2D(6, 7)(inputs)
        x = layers.BatchNormalization()(x)
        outputs = layers.Activation('swish')(x)
        return keras.Model(inputs=inputs, outputs=[outputs, y, z, w])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(3, 4)(inputs)
        x = layers.Conv2D(3, 4)(x)
        x1, y1, z1, w1 = self.inner_functional_model(x.shape)(x)
        x2, y2, z2, w2 = self.inner_functional_model(x.shape)(x)
        model = keras.Model(inputs=inputs, outputs=[z2, x1, y1, z1, w1, x2, y2, w2])
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        num_conv_layers = len([x for x in quantized_model.layers if isinstance(x, layers.Conv2D)])
        self.unit_test.assertTrue(num_conv_layers == 10)
        for l in quantized_model.layers:
            if hasattr(l, 'layer'):
                self.unit_test.assertFalse(isinstance(l.layer, Functional) or isinstance(l.layer, Sequential))
        self.unit_test.assertTrue(len(quantized_model.output) == len(float_model.output))
        for qo, fo in zip(quantized_model.output, float_model.output):
            self.unit_test.assertTrue(qo.shape.as_list() == fo.shape.as_list())
