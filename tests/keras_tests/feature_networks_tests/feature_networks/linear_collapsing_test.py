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

from abc import ABC
import model_compression_toolkit as mct
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


class BaseConv2DCollapsingTest(BaseKerasFeatureNetworkTest, ABC):

    def __init__(self, unit_test):
        super(BaseConv2DCollapsingTest, self).__init__(unit_test=unit_test, input_shape=(32,32,16))

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
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        self.unit_test.assertTrue(len(quantized_model.layers) < len(float_model.layers), msg=f'fail number of layers should decrease!')
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class TwoConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', bias_initializer='glorot_uniform')(inputs)
        y = layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        super().compare(quantized_model, float_model, input_x, quantization_info)
        for layer in quantized_model.layers:
            if type(layer) == Conv2D:
                self.unit_test.assertTrue(len(layer.weights) == 2, msg=f'fail Bias should appear in weights!!')

class ThreeConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        y = layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, activation='relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        super().compare(quantized_model, float_model, input_x, quantization_info)
        for layer in quantized_model.layers:
            if type(layer) == Conv2D:
                self.unit_test.assertTrue(len(layer.weights) == 1,msg=f'fail Bias should not appear in weights!!')


class FourConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', bias_initializer='glorot_uniform', activation='linear')(inputs)
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform', activation='linear')(x)
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', bias_initializer='glorot_uniform', activation='linear')(x)
        y = layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform', activation='relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        super().compare(quantized_model, float_model, input_x, quantization_info)
        for layer in quantized_model.layers:
            if type(layer) == Conv2D:
                self.unit_test.assertTrue(len(layer.weights) == 2,msg=f'fail Bias should appear in weights!!')

class SixConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform')(x)
        x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform',  activation='relu')(x)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform')(x)
        y = layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform', activation='relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        super().compare(quantized_model, float_model, input_x, quantization_info)
        for layer in quantized_model.layers:
            if type(layer) == Conv2D:
                self.unit_test.assertTrue(len(layer.weights) == 2,msg=f'fail Bias should appear in weights!!')