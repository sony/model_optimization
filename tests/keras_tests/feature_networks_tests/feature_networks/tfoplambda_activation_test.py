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
from model_compression_toolkit.keras.graph_substitutions.substitutions.shift_negative_activation import SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS

keras = tf.keras
layers = keras.layers


class TFOpLambdaActivationTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test, activation_function, shift_negative_activation_correction=False):
        self.activation_function = activation_function
        self.shift_negative_activation_correction = shift_negative_activation_correction
        super().__init__(unit_test, val_batch_size=1)

    def get_quantization_config(self):
        return mct.QuantizationConfig(weights_n_bits=16,
                                      activation_n_bits=14,
                                      weights_bias_correction=False,
                                      shift_negative_activation_correction=self.shift_negative_activation_correction,
                                      shift_negative_ratio=1.0,
                                      activation_channel_equalization=True,
                                      )

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2D(3, 4)(inputs)
        x = self.activation_function(x)
        outputs = layers.Conv2D(3, 4)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check no quantization between conv and activation and same activation function
        self.unit_test.assertTrue(quantized_model.layers[3].function is self.activation_function)
        self.unit_test.assertTrue(quantized_model.layers[4].function is tf.quantization.fake_quant_with_min_max_vars)
        if self.shift_negative_activation_correction:
            self.unit_test.assertTrue(quantized_model.layers[4]._inbound_nodes[0].call_kwargs['num_bits'] == SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS)
            self.unit_test.assertTrue(quantized_model.layers[5].node_def.op == 'Add')
            self.unit_test.assertTrue(quantized_model.layers[6].function is tf.quantization.fake_quant_with_min_max_vars)
            self.unit_test.assertTrue(quantized_model.layers[6]._inbound_nodes[0].call_kwargs['num_bits'] == 14)
        else:
            self.unit_test.assertTrue(quantized_model.layers[4]._inbound_nodes[0].call_kwargs['num_bits'] == 14)

        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'{self.activation_function}: fail cosine similarity check: {cs}')
