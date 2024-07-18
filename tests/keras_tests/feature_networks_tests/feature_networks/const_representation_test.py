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
import tensorflow as tf
import numpy as np

import model_compression_toolkit as mct
from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


class ConstRepresentationTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test, layer, const, is_list_input=False, input_reverse_order=False, use_kwargs=False,
                 input_shape=(32, 32, 16)):
        super(ConstRepresentationTest, self).__init__(unit_test=unit_test, input_shape=input_shape)
        self.layer = layer
        self.const = const
        self.is_list_input = is_list_input
        self.input_reverse_order = input_reverse_order
        self.use_kwargs = use_kwargs

    def generate_inputs(self):
        # need positive inputs so won't divide with zero or take root of negative number
        return [1 + np.random.random(in_shape) for in_shape in self.get_input_shapes()]

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 16,
                                     'activation_n_bits': 16,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="const_representation_test", tp_model=tp)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = inputs
        if self.is_list_input:
            if self.input_reverse_order:
                x = self.layer([self.const, x])
            else:
                x = self.layer([x, self.const])
        else:
            if self.input_reverse_order:
                if self.use_kwargs:
                    x = self.layer(x=self.const, y=x)
                else:
                    x = self.layer(self.const, x)
            else:
                if self.use_kwargs:
                    x = self.layer(x=x, y=self.const)
                else:
                    x = self.layer(x, self.const)
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}, for operator {self.layer}')


class ConstRepresentationListTypeArgsTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test, input_shape=(32, 32, 16)):
        super(ConstRepresentationListTypeArgsTest, self).__init__(unit_test=unit_test, input_shape=input_shape)

    def generate_inputs(self):
        # need positive inputs so won't divide with zero or take root of negative number
        return [1 + np.random.random(in_shape) for in_shape in self.get_input_shapes()]

    def get_tpc(self):
        return get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.image.resize(inputs, size=self.get_input_shapes()[0][1:3])
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class ConstRepresentationMatMulTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, input_shape=(32, 32, 16)):
        super(ConstRepresentationMatMulTest, self).__init__(unit_test=unit_test, input_shape=input_shape)

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 16,
                                     'activation_n_bits': 16,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="const_representation_test", tp_model=tp)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        c = (np.ones((16, 8)) + np.random.random((16, 8))).astype(np.float32)
        x = inputs
        x1 = tf.matmul(x, c)
        x2 = tf.matmul(x, b=c)
        x3 = tf.matmul(a=x, b=c)

        transpose_c = tf.transpose(c, perm=[1, 0])
        x4 = tf.matmul(x, transpose_c, False, True)
        x5 = tf.matmul(x, b=transpose_c, transpose_b=True)
        x6 = tf.matmul(a=x, b=transpose_c, transpose_b=True)

        x = x1 + x2 + x3 + x4 + x5 + x6
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class ConstRepresentationMultiInputTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test, input_shape=(32, 32, 16)):
        super(ConstRepresentationMultiInputTest, self).__init__(unit_test=unit_test, input_shape=input_shape)

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 16,
                                     'activation_n_bits': 16,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="const_representation_test", tp_model=tp)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Concatenate()([inputs, np.random.random((1, 32, 32, 3)), inputs, np.random.random((1, 32, 32, 3))])
        x1 = layers.Add()([np.random.random((1, x.shape[-1])), x, np.random.random((1, x.shape[-1]))])
        x2 = layers.Multiply()([x, np.random.random((1, x.shape[-1])), x, np.random.random((1, x.shape[-1]))])
        x3 = tf.add_n([x1, np.random.random(x.shape.as_list()).astype(np.float32), x2])
        x = tf.concat([x1, x2, np.random.random(x3.shape.as_list()).astype(np.float32), x3], 1)
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')
