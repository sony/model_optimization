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

from model_compression_toolkit.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf

from tests.common_tests.base_layer_test import LayerTestMode
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.tpc_keras import get_16bit_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


def update_kernel_for_bn_folding_fn(conv_layer: layers.Conv2D,
                                    kernel: np.ndarray,
                                    weights_scale):
    """
    Args:
        conv_layer: Convolution layer to update the weights.
        kernel: The Convolution layer's weights.
        weights_scale: Weights scale factor to multiply the conv layer's weights by.

    Returns:
        The modified convolution layer's weights.
    """
    if conv_layer.__class__ == layers.DepthwiseConv2D:
        kernel = kernel * weights_scale.reshape(1, 1, kernel.shape[-2], kernel.shape[-1])
    elif conv_layer.__class__ == layers.Conv2DTranspose:
        kernel = kernel * weights_scale.reshape(1, 1, -1, 1)
    else:
        kernel = kernel * weights_scale.reshape(1, 1, 1, -1)
    return kernel


class BaseBatchNormalizationFolding(BaseKerasFeatureNetworkTest, ABC):

    def __init__(self, unit_test):
        super(BaseBatchNormalizationFolding, self).__init__(unit_test=unit_test, experimental_exporter=True)

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 16,
                                     'activation_n_bits': 16,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="bn_folding_test", tp_model=tp)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING,
                                      mct.QuantizationErrorMethod.NOCLIPPING,
                                      False, False, True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check the conv weights after the bn folding
        float_conv = float_model.layers[1]

        if float_conv.__class__ == layers.SeparableConv2D:
            float_kernel = float_conv.weights[1]
            float_bias = float_conv.weights[2]

            quant_conv = quantized_model.layers[3]
        else:
            float_kernel = float_conv.weights[0]
            float_bias = float_conv.weights[1]

            quant_conv = quantized_model.layers[2]

        attr = 'depthwise_kernel' if isinstance(quant_conv.layer, layers.DepthwiseConv2D) else 'kernel'
        quant_kernel = getattr(quant_conv.layer, attr)
        quant_bias = quant_conv.layer.bias

        float_bn = float_model.layers[2]
        float_gamma = float_bn.weights[0]
        float_beta = float_bn.weights[1]
        float_moving_mean = float_bn.weights[2]
        float_moving_variance = float_bn.weights[3]
        float_epsilon = float_bn.epsilon

        weights_scale = float_gamma / np.sqrt(float_moving_variance + float_epsilon)
        bias = float_beta + (float_bias - float_moving_mean) * weights_scale
        kernel = update_kernel_for_bn_folding_fn(conv_layer=float_conv, kernel=float_kernel.numpy(),
                                                 weights_scale=weights_scale.numpy())

        self.unit_test.assertTrue(np.all(quant_kernel.numpy() == kernel))
        self.unit_test.assertTrue(np.all(quant_bias.numpy() == bias))

        # check for no bn after the bn folding
        self.unit_test.assertFalse(layers.BatchNormalization in
                                   [layer.__class__ for layer in quantized_model.layers])

        # check the output didn't change
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class Conv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(2, 3, padding='same')(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class Conv2DBNConcatnFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(2, 3, padding='same')(inputs)
        x_bn = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x_bn)
        x2 = layers.Activation('tanh')(x_bn)
        x = layers.Concatenate()([x, x_bn, x2])

        return tf.keras.models.Model(inputs=inputs, outputs=x)


class Conv2DTransposeBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2DTranspose(2, 3, padding='same')(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class DepthwiseConv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(1, padding='same')(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class DepthwiseConv2DBNFoldingHighMultiplierTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(1, padding='same', depth_multiplier=3)(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class SeparableConv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.SeparableConv2D(1, 3, padding='same')(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)
