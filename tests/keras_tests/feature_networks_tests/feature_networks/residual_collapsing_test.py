# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import model_compression_toolkit as mct
import tensorflow as tf
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


class BaseResidualCollapsingTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test):
        super(BaseResidualCollapsingTest, self).__init__(unit_test=unit_test, input_shape=(16,16,3))

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 32,
                                     'activation_n_bits': 32,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="linear_collapsing_test", tp_model=tp)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING, mct.QuantizationErrorMethod.NOCLIPPING,
                                      False, False, True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'fail: out shape is not as expected!')
        for layer in quantized_model.layers:
            self.unit_test.assertFalse(type(layer) == layers.Add or type(layer) == tf.add, msg=f'fail: add residual is still in the model')
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail: cosine similarity check:{cs}')


class ResidualCollapsingTest1(BaseResidualCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same', bias_initializer='glorot_uniform')(inputs)
        y = layers.Add()([x, inputs])
        y = layers.ReLU()(y)
        return tf.keras.models.Model(inputs=inputs, outputs=y)


class ResidualCollapsingTest2(BaseResidualCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=3, kernel_size=(3, 4), padding='same', bias_initializer='glorot_uniform')(inputs)
        x1 = layers.Add()([x, inputs])
        x2 = layers.Conv2D(filters=3, kernel_size=(2, 2), padding='same', bias_initializer='glorot_uniform')(x1)
        x3 = tf.add(x2, x1)
        x3 = layers.ReLU()(x3)
        x4 = layers.Conv2D(filters=3, kernel_size=(1, 3), padding='same')(x3)
        y = layers.Add()([x3, x4])
        y = layers.ReLU()(y)
        return tf.keras.models.Model(inputs=inputs, outputs=y)