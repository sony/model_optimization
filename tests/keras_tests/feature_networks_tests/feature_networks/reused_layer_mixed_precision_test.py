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


from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from tests.keras_tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class ReusedLayerMixedPrecisionTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE,
                                    mct.ThresholdSelectionMethod.MSE,
                                    mct.QuantizationMethod.POWER_OF_TWO,
                                    mct.QuantizationMethod.POWER_OF_TWO,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    activation_channel_equalization=True,
                                    relu_unbound_correction=True,
                                    input_scaling=True,
                                    activation_n_bits=16)

        return MixedPrecisionQuantizationConfig(qc, weights_n_bits=[2, 16, 4])

    def create_inputs_shape(self):
        return [[self.val_batch_size, 224, 244, 3]]

    def create_feature_network(self, input_shape):
        layer = layers.Conv2D(3, 4)
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layer(inputs)
        y = layer(inputs)
        x = layers.BatchNormalization()(x)
        y = layers.BatchNormalization()(y)
        x = layers.ReLU()(x)
        x = layers.Add()([x, y])
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def get_kpi(self):
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class ReusedSeparableMixedPrecisionTest(ReusedLayerMixedPrecisionTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        layer = layers.SeparableConv2D(3, 3, padding='same')
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layer(inputs)
        y = layer(inputs)
        x = layers.BatchNormalization()(x)
        y = layers.BatchNormalization()(y)
        x = layers.ReLU()(x)
        x = layers.Add()([x, y])
        model = keras.Model(inputs=inputs, outputs=x)
        return model
