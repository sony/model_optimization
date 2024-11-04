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
from model_compression_toolkit.core import QuantizationConfig
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

import tensorflow as tf
import numpy as np

from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers

"""
This test checks the Activation Bias Correction feature.
"""

class BaseActivationBiasCorrectionTest(BaseKerasFeatureNetworkTest):
    """
    This test checks the Activation Bias Correction feature.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return QuantizationConfig(weights_bias_correction=False,
                                  weights_second_moment_correction=False,
                                  activation_bias_correction=True)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Activation('gelu')(inputs)
        x = layers.Dropout(0.5)(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        float_dense_layers = get_layers_from_model_by_type(float_model, layers.Dense)
        quantized_dense_layers = get_layers_from_model_by_type(quantized_model, layers.Dense)

        bias = float_dense_layers[-1].bias
        bias_after_activation_bias_correction = quantized_dense_layers[-1].layer.bias

        self.unit_test.assertFalse(np.array_equal(bias, bias_after_activation_bias_correction),
                                   msg=f"Error in activation bias correction: expected a change in the bias value.")


class BaseActivationBiasCorrectionConvTest(BaseKerasFeatureNetworkTest):
    """
    This test checks the Activation Bias Correction feature with Conv2D layer.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        # A small value set to the activation bias correction threshold only to activate the threshold
        # filtering without changing the bias correction values.
        return QuantizationConfig(weights_bias_correction=False,
                                  weights_second_moment_correction=False,
                                  activation_bias_correction=True,
                                  activation_bias_correction_threshold=1e-6)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Activation('swish')(inputs)
        x = layers.ZeroPadding2D(2)(x)
        outputs = layers.Conv2D(filters=3, kernel_size=1, use_bias=True)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        float_conv_layers = get_layers_from_model_by_type(float_model, layers.Conv2D)
        quantized_conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)

        bias = float_conv_layers[-1].bias
        bias_after_activation_bias_correction = quantized_conv_layers[-1].layer.bias

        self.unit_test.assertFalse(np.array_equal(bias, bias_after_activation_bias_correction),
                                   msg=f"Error in activation bias correction: expected a change in the bias value.")


class BaseActivationBiasCorrectionDWConvTest(BaseKerasFeatureNetworkTest):
    """
    This test checks the Activation Bias Correction feature with DepthWiseConv2D layer.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return QuantizationConfig(weights_bias_correction=False,
                                  weights_second_moment_correction=False,
                                  activation_bias_correction=True)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.nn.gelu(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        outputs = layers.DepthwiseConv2D(kernel_size=1, use_bias=True, bias_initializer='glorot_uniform',
                                         depth_multiplier=1)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        float_dw_conv_layers = get_layers_from_model_by_type(float_model, layers.DepthwiseConv2D)
        quantized_dw_conv_layers = get_layers_from_model_by_type(quantized_model, layers.DepthwiseConv2D)

        bias = float_dw_conv_layers[-1].bias
        bias_after_activation_bias_correction = quantized_dw_conv_layers[-1].layer.bias

        self.unit_test.assertFalse(np.array_equal(bias, bias_after_activation_bias_correction),
                                   msg=f"Error in activation bias correction: expected a change in the bias value.")


class BaseActivationBiasCorrectionReshapeConvTest(BaseKerasFeatureNetworkTest):
    """
    This test checks the Activation Bias Correction feature.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        # A large value is assigned to the activation bias correction threshold to enable threshold filtering,
        # which adjusts the bias correction values to zero.
        return QuantizationConfig(weights_bias_correction=False,
                                  weights_second_moment_correction=False,
                                  activation_bias_correction=True,
                                  activation_bias_correction_threshold=1e9)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Activation('swish')(inputs)
        x = layers.Flatten()(x)
        x = layers.Reshape(target_shape=(8, 8, 3))(x)
        outputs = layers.Conv2D(filters=3, kernel_size=1, use_bias=True, bias_initializer='ones')(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        float_conv_layers = get_layers_from_model_by_type(float_model, layers.Conv2D)
        quantized_conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)

        bias = float_conv_layers[-1].bias
        bias_after_activation_bias_correction = quantized_conv_layers[-1].layer.bias

        self.unit_test.assertTrue(np.array_equal(bias, bias_after_activation_bias_correction),
                                  msg=f"Error in activation bias correction: expected no change in the bias value in "
                                      f"case of activation_bias_correction_threshold 1e9.")
