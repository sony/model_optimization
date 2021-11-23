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


from tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf
import numpy as np
from model_compression_toolkit.keras.back2framework.model_builder import is_layer_fake_quant
from tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class BaseInputScalingTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING, mct.ThresholdSelectionMethod.NOCLIPPING,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO,
                                      16, 16, False, False, True, input_scaling=True)


    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(is_layer_fake_quant(quantized_model.layers[1]))
        self.unit_test.assertTrue(quantization_info.input_scale != 1)
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class InputScalingDenseTest(BaseInputScalingTest):
    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Dense(20)(inputs)
        x = layers.ReLU()(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)


class InputScalingConvTest(BaseInputScalingTest):
    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2D(2, 3, padding='same')(inputs)
        x = layers.ReLU()(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)


class InputScalingDWTest(BaseInputScalingTest):
    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.DepthwiseConv2D(1, padding='same')(inputs)
        x = layers.ReLU()(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)


class InputScalingZeroPadTest(BaseInputScalingTest):
    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.ZeroPadding2D()(inputs)
        x = layers.DepthwiseConv2D(1, padding='same')(x)
        x = layers.ReLU()(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)
