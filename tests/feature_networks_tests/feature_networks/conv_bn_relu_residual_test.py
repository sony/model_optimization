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
from tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class ConvBnReluResidualTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING, mct.ThresholdSelectionMethod.NOCLIPPING,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO,
                                      16, 16, True, True, True)


    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        y = layers.Conv2D(7, 8)(inputs)
        x = layers.BatchNormalization()(y)
        x = layers.Activation('relu')(x)
        outputs = layers.Add()([x, y])
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(isinstance(quantized_model.layers[4], layers.BatchNormalization))
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')
