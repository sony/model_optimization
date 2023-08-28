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

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf

from tests.common_tests.base_layer_test import LayerTestMode
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.tpc_keras import get_16bit_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity, normalized_mse
from tests.keras_tests.utils import get_layers_from_model_by_type

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

    def __init__(self, unit_test, linear_layer):
        self.linear_layer = linear_layer
        super(BaseBatchNormalizationFolding, self).__init__(unit_test=unit_test, experimental_exporter=True)

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 16,
                                     'activation_n_bits': 16,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="bn_folding_test", tp_model=tp)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.NOCLIPPING,
                                      mct.core.QuantizationErrorMethod.NOCLIPPING,
                                      False, False, True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check the conv weights after the bn folding
        float_conv = float_model.layers[1]

        if float_conv.__class__ == layers.SeparableConv2D:
            float_kernel = float_conv.weights[1]
            float_bias = float_conv.weights[2]

            quant_conv = get_layers_from_model_by_type(quantized_model, layers.Conv2D)[0]
        else:
            float_kernel = float_conv.weights[0]
            float_bias = float_conv.weights[1]

            quant_conv = get_layers_from_model_by_type(quantized_model, self.linear_layer)[0]

        attr = 'depthwise_kernel' if isinstance(quant_conv, layers.DepthwiseConv2D) else 'kernel'
        quant_kernel = getattr(quant_conv, attr)
        quant_bias = quant_conv.bias

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
        super().__init__(unit_test,
                         linear_layer=layers.Conv2D)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = self.linear_layer(2, 3, padding='same')(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class Conv2DBNConcatnFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test,
                         linear_layer=layers.Conv2D)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = self.linear_layer(2, 3, padding='same')(inputs)
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
        super().__init__(unit_test,
                         linear_layer=layers.Conv2DTranspose)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = self.linear_layer(2, 3, padding='same')(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class DepthwiseConv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test,
                         linear_layer=layers.DepthwiseConv2D)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = self.linear_layer(1, padding='same')(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class DepthwiseConv2DBNFoldingHighMultiplierTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test,
                         linear_layer=layers.DepthwiseConv2D)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = self.linear_layer(1, padding='same', depth_multiplier=3)(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class SeparableConv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test,
                         linear_layer=layers.SeparableConv2D)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = self.linear_layer(1, 3, padding='same')(inputs)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones")(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class BNForwardFoldingTest(BaseKerasFeatureNetworkTest):
    """
    This test checks the BatchNorm forward folding feature. When conversion_applied is False
    test that the BN isn't folded
    """
    def __init__(self, unit_test, test_layer, conversion_applied, add_bn=False, is_dwconv=False):
        super().__init__(unit_test=unit_test, experimental_exporter=True, val_batch_size=2)
        self.test_layer = test_layer
        self.conversion_applied = conversion_applied
        self.add_bn = add_bn
        self.is_dwconv = is_dwconv

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        if self.is_dwconv:
            x = layers.DepthwiseConv2D(1, bias_initializer='glorot_uniform')(inputs)
        else:
            x = layers.BatchNormalization(beta_initializer='glorot_uniform',
                                          gamma_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1.05),
                                          moving_mean_initializer='glorot_uniform',
                                          moving_variance_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1.05)
                                          )(inputs)
        x = self.test_layer(x)
        if self.add_bn:
            x = layers.BatchNormalization(beta_initializer='glorot_uniform',
                                          gamma_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1.05),
                                          moving_mean_initializer='glorot_uniform',
                                          moving_variance_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1.05)
                                          )(x)
        x = layers.Activation('tanh')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 16,
                                     'activation_n_bits': 16,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="bn_folding_test", tp_model=tp)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        if self.is_dwconv:
            is_bn_in_model = (sum([isinstance(l, tf.keras.layers.DepthwiseConv2D) for l in float_model.layers]) ==
                              sum([isinstance(l, tf.keras.layers.DepthwiseConv2D) for l in quantized_model.layers]))
        else:
            is_bn_in_model = any([isinstance(l, tf.keras.layers.BatchNormalization) for l in quantized_model.layers])

        self.unit_test.assertTrue(self.conversion_applied is not is_bn_in_model)

        # Checking on multiple inputs to reduce probability for numeric error that will randomly fail the test
        self.unit_test.assertEqual(input_x[0].shape[0], 2, "Expecting batch of size 2 for BN folding test.")

        out_float = float_model(input_x)
        out_quant = quantized_model(input_x)

        norm_mse, _, max_error, _ = normalized_mse(out_float.numpy(), out_quant.numpy())

        self.unit_test.assertTrue(np.isclose(norm_mse[0], 0, atol=1e-5) or np.isclose(norm_mse[1], 0, atol=1e-5))
        self.unit_test.assertTrue(np.isclose(max_error[0], 0, atol=1e-4) or np.isclose(max_error[1], 0, atol=1e-4))

