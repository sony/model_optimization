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
from model_compression_toolkit.core.keras.constants import KERNEL_SIZE
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

import tensorflow as tf
import numpy as np

from tests.keras_tests.utils import get_layers_from_model_by_type
from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import get_tpc

keras = tf.keras
layers = keras.layers

"""
This test checks the Activation Bias Correction feature.
"""


class BaseActivationBiasCorrectionTest(BaseKerasFeatureNetworkTest):
    """
    This test checks the Activation Bias Correction feature.
    """

    def __init__(self, unit_test,
                 prev_layer,
                 bypass_layer_list,
                 linear_layer,
                 activation_bias_correction_threshold=0.0):
        super().__init__(unit_test)
        self.prev_layer = prev_layer
        self.bypass_layer_list = bypass_layer_list
        self.linear_layer = linear_layer
        self.activation_bias_correction_threshold = activation_bias_correction_threshold

    def get_quantization_config(self):
        return QuantizationConfig(weights_bias_correction=False,
                                  weights_second_moment_correction=False,
                                  shift_negative_activation_correction=False,
                                  activation_bias_correction=True,
                                  activation_bias_correction_threshold=self.activation_bias_correction_threshold)

    def get_tpc(self):
        tpc = get_tpc()
        return tpc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = self.prev_layer(inputs)

        for bypass_layer in self.bypass_layer_list:
            x = bypass_layer(x)

        outputs = self.linear_layer(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        float_linear_layers = get_layers_from_model_by_type(float_model, type(self.linear_layer))
        quantized_linear_layers = get_layers_from_model_by_type(quantized_model, type(self.linear_layer))

        bias = float_linear_layers[-1].bias
        bias_after_activation_bias_correction = quantized_linear_layers[-1].layer.bias

        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)

        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')

        if getattr(float_linear_layers[-1], KERNEL_SIZE, None) in [None, 1, (1, 1)]:
            if self.activation_bias_correction_threshold > 1e8:
                self.unit_test.assertTrue(np.array_equal(bias, bias_after_activation_bias_correction),
                                          msg=f"Error in activation bias correction: expected no change in the bias "
                                              f"value in case of activation_bias_correction_threshold "
                                              f"{self.activation_bias_correction_threshold}.")

            else:
                self.unit_test.assertFalse(np.array_equal(bias, bias_after_activation_bias_correction),
                                           msg=f"Error in activation bias correction: expected a change in the bias "
                                               f"value.")
        else:
            self.unit_test.assertTrue(np.array_equal(bias, bias_after_activation_bias_correction),
                                      msg=f"Error in activation bias correction: expected no change in the bias value "
                                          f"in case of conv with kernel different than 1 or (1, 1).")
