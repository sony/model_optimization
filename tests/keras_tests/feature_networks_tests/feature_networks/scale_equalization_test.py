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


import numpy as np
import tensorflow as tf
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

import model_compression_toolkit as mct
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class ScaleEqualizationTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, first_op2d, second_op2d, mid_activation=False, second_op2d_zero_pad=False):
        self.first_op2d = first_op2d
        self.second_op2d = second_op2d
        self.mid_act = mid_activation
        self.second_op2d_zero_pad = second_op2d_zero_pad
        super().__init__(unit_test,
                         input_shape=(16,16,3))

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE,
                                      mct.ThresholdSelectionMethod.MSE,
                                      mct.QuantizationMethod.POWER_OF_TWO,
                                      mct.QuantizationMethod.POWER_OF_TWO,
                                      16,
                                      16,
                                      relu_unbound_correction=False,
                                      weights_bias_correction=False,
                                      weights_per_channel_threshold=True,
                                      activation_channel_equalization=True)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = self.first_op2d(inputs)
        if self.mid_act:
            x = layers.ReLU()(x)
        if self.second_op2d_zero_pad:
            x = layers.ZeroPadding2D()(x)
        outputs = self.second_op2d(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        q_first_linear_op_index = 2
        q_second_linear_op_index = 4 + int(self.second_op2d_zero_pad) + int(self.mid_act) + int(isinstance(self.first_op2d, (layers.Dense, layers.Conv2DTranspose)) and self.mid_act)
        f_first_linear_op_index = 1
        f_second_linear_op_index = 2 + int(self.second_op2d_zero_pad) + int(self.mid_act)


        alpha_nonzero_index = quantized_model.layers[q_first_linear_op_index].weights[0].numpy() != 0
        alpha = ((quantized_model.layers[q_first_linear_op_index].weights[0]) / (float_model.layers[f_first_linear_op_index].weights[0])).numpy()
        alpha = alpha[alpha_nonzero_index].mean()

        beta_nonzero_index = quantized_model.layers[q_second_linear_op_index].weights[0].numpy() != 0
        beta = ((float_model.layers[f_second_linear_op_index].weights[0]) / (quantized_model.layers[q_second_linear_op_index].weights[0])).numpy()
        beta = beta[beta_nonzero_index].mean()

        self.unit_test.assertTrue(np.allclose(alpha, beta, atol=1e-1))

