# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
from tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf
import numpy as np
from tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class BaseBatchNormalizationFolding(BaseFeatureNetworkTest, ABC):
    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING, mct.ThresholdSelectionMethod.NOCLIPPING,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO,
                                      16, 16, False, False, True, enable_weights_quantization=False,
                                      enable_activation_quantization=False)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class Conv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2D(2, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class Conv2DBNConcatnFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2D(2, 3, padding='same')(inputs)
        x_bn = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x_bn)
        x2 = layers.Activation('tanh')(x_bn)
        x = layers.Concatenate()([x, x_bn, x2])

        return tf.keras.models.Model(inputs=inputs, outputs=x)


class Conv2DTransposeBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2DTranspose(2, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class DepthwiseConv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.DepthwiseConv2D(1, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class DepthwiseConv2DBNFoldingHighMultiplierTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.DepthwiseConv2D(1, padding='same', depth_multiplier=3)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class SeparableConv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.SeparableConv2D(1, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)
