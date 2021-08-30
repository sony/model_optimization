# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


from abc import ABC
from tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import sony_model_optimization_package as smop
import tensorflow as tf
import numpy as np
from tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class BaseBatchNormalizationFolding(BaseFeatureNetworkTest, ABC):
    def get_quantization_config(self):
        return smop.QuantizationConfig(smop.ThresholdSelectionMethod.NOCLIPPING, smop.ThresholdSelectionMethod.NOCLIPPING,
                                       smop.QuantizationMethod.SYMMETRIC_UNIFORM, smop.QuantizationMethod.SYMMETRIC_UNIFORM,
                                       16, 16, False, False, True, enable_weights_quantization=False,
                                       enable_activation_quantization=False)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 16, 16, 1]]

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class Conv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test, num_calibration_iter=1, val_batch_size=32)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2D(2, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class Conv2DBNConcatnFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test, num_calibration_iter=1, val_batch_size=32)

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
        super().__init__(unit_test, num_calibration_iter=1, val_batch_size=32)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2DTranspose(2, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class DepthwiseConv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test, num_calibration_iter=1, val_batch_size=32)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.DepthwiseConv2D(1, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class DepthwiseConv2DBNFoldingHighMultiplierTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test, num_calibration_iter=1, val_batch_size=32)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 16, 16, 2]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.DepthwiseConv2D(1, padding='same', depth_multiplier=3)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)

class SeparableConv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test, num_calibration_iter=1, val_batch_size=32)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.SeparableConv2D(1, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)
