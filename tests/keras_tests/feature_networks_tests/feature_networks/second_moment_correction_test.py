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

import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.keras.constants import EPSILON_VAL
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers
tp = mct.target_platform

# TODO:
# DepthwiseConv2D with channels test
# SeperableConv test


class BaseSecondMomentTest(BaseKerasFeatureNetworkTest, ABC):
    """
    This is the base test for the Second Moment Correction feature.
    """
    def __init__(self, unit_test):
        super(BaseSecondMomentTest, self).__init__(unit_test=unit_test, val_batch_size=128, input_shape=(32, 32, 1))

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 16,
                                     'activation_n_bits': 16,
                                     'weights_quantization_method': QuantizationMethod.SYMMETRIC})
        return generate_keras_tpc(name="second_moment_correction_test", tp_model=tp)

    def get_quantization_config(self):
        return mct.QuantizationConfig(weights_second_moment_correction=True, weights_second_moment_iters=200)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        quantized_model_kernel = quantized_model.layers[2].weights[0]
        quantized_model_bias = quantized_model.layers[2].weights[1]
        float_model_gamma = float_model.layers[2].weights[0]
        float_model_beta = float_model.layers[2].weights[1]
        float_model_kernel = float_model.layers[1].weights[0]
        float_model_bias = float_model.layers[1].weights[1]
        input_var = np.var(self.inp)
        input_mean = np.mean(self.inp)
        eps = EPSILON_VAL
        weight_scale = np.sqrt(float_model_gamma + eps) / np.sqrt(input_var + eps)

        # new_kernel = kernel * gamma/sqrt(moving_var+eps)
        # new_bias = beta + (bias - moving_mean) * *gamma/sqrt(moving_var+eps)
        calculated_kernel = float_model_kernel * weight_scale
        calculated_bias = float_model_beta + (float_model_bias - input_mean) * weight_scale

        self.unit_test.assertTrue(np.isclose(quantized_model_kernel, calculated_kernel, atol=1e-1))
        self.unit_test.assertTrue(np.isclose(quantized_model_bias, calculated_bias, atol=1e-1))

    def generate_inputs(self):
        # We want to keep the same input in order to stabilize the input's statistics
        if self.i == 0:
            self.inp = [np.random.normal(scale=0.5, loc=8.0, size=in_shape) for in_shape in self.get_input_shapes()]
            self.i += 1
        return self.inp


class DepthwiseConv2DSecondMomentTest(BaseSecondMomentTest):
    """
    This is the test for the Second Moment Correction feature with DepthwiseConv2D.
    """
    def __init__(self, unit_test):
        self.i = 0
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D((1, 1), padding='same',
                                   depthwise_initializer='ones',
                                   kernel_initializer="ones",
                                   bias_initializer="zeros")(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class Conv2DSecondMomentTest(BaseSecondMomentTest):
    """
    This is the test for the Second Moment Correction feature with Conv2d.
    """
    def __init__(self, unit_test):
        self.i = 0
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(1, 1, padding='same',
                          kernel_initializer="ones",
                          bias_initializer="zeros")(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class Conv2DTSecondMomentTest(BaseSecondMomentTest):
    """
    This is the test for the Second Moment Correction feature with Conv2DTranspose.
    """
    def __init__(self, unit_test):
        self.i = 0
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2DTranspose(1, 1, padding='same',
                                   kernel_initializer="ones",
                                   bias_initializer="zeros")(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)
