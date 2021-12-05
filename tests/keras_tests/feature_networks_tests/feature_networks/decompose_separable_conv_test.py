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


from tests.keras_tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class DecomposeSeparableConvTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test, depth=1):
        self.depth_multiplier = depth
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING, mct.ThresholdSelectionMethod.NOCLIPPING,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO,
                                      16, 16, weights_bias_correction=False,
                                      weights_per_channel_threshold=True, enable_activation_quantization=True,
                                      enable_weights_quantization=True, relu_unbound_correction=False)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 4, 5]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        outputs = layers.SeparableConv2D(1, 2, depth_multiplier=self.depth_multiplier)(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantized_model.layers) == 6)
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2], layers.DepthwiseConv2D))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[4], layers.Conv2D))
        self.unit_test.assertTrue(quantized_model.layers[2].weights[0].shape == (2, 2, 5, 1 * self.depth_multiplier))
        self.unit_test.assertTrue(quantized_model.layers[4].weights[0].shape == (1, 1, 5 * self.depth_multiplier, 1))
        self.unit_test.assertTrue(quantized_model.output_shape == float_model.output_shape)

        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')
